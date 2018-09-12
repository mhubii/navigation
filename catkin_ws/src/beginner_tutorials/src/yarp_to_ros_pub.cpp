#include <iostream>
#include <stdio.h>
#include <vector>
#include <signal.h>

#include <yarp/os/all.h>
#include <yarp/dev/all.h>
#include <yarp/sig/all.h>
#include <yarp/rosmsg/sensor_msgs/Imu.h>
#include <yarp/rosmsg/sensor_msgs/Image.h>
#include <yarp/rosmsg/impl/yarpRosHelper.h>

// A port that stores the value last received.
class ImuPort : public yarp::os::Port
{
public:

	ImuPort() {

		setCallbackLock(&mutex_);
		setReader(last_);
	}

	// Return last received value.
	yarp::sig::VectorOf<yarp::os::NetFloat64> LastRead() {

		lockCallback();
		yarp::sig::VectorOf<yarp::os::NetFloat64> tmp = last_;
		unlockCallback();

		return tmp;		
	}

private:

	yarp::sig::VectorOf<yarp::os::NetFloat64> last_;
	yarp::os::Mutex mutex_;
};


// YarpToRosImg implements a yarp::os::RateThread that
// reads data from a remote camera device and publishes 
// it to a Ros topic.
class YarpToRosImg : public yarp::os::RateThread
{
public:

	YarpToRosImg(int period, std::string local, std::string remote, std::string topic) : yarp::os::RateThread(period), local_(local), remote_(remote), topic_(topic) {	};

	// Call run every period milliseconds.
	virtual void run() {

		// Get image.
		fg_->getImage(rgb_);
		mono_.copy(rgb_, rgb_.width(), rgb_.height());

		// Publish to ROS.		
		yarp::rosmsg::sensor_msgs::Image& msg = publisher_.prepare();

		// Header
		msg.header.seq = msg_count_++;
		msg.header.stamp = yarp::os::Time::now();
		msg.header.frame_id = "chassis";

		// Size.
		msg.height = mono_.height();
		msg.width = mono_.width();

		// Encoding.
		msg.encoding = "mono8";
		msg.is_bigendian = 0;
		msg.step = mono_.getRowSize();

		// Data.
		std::vector<std::uint8_t> data(mono_.getRawImage(), mono_.getRawImage() + mono_.getRawImageSize());
		msg.data = data;

		publisher_.write();
	};

	// Initialize the thread. On false return, thread doesn't call run and exits.
	virtual bool threadInit() {

		// Publisher.
		if (!publisher_.topic(topic_)) {

			std::cerr << "Could not publish to topic " << topic_ << std::endl;
			std::exit(1);
		};

		// Device driver.
		options_.put("device", "remote_grabber");
		options_.put("local", local_);
		options_.put("remote", remote_);

		pd_ = new yarp::dev::PolyDriver(options_);

		if (!pd_->isValid()) {

			std::cerr << "Device driver remote_grabber not available." << std::endl;
			std::exit(1);
		}

		if (!pd_->view(fg_)) {

			std::cerr << "Coult not acquire interface." << std::endl;
			std::exit(1);
		};

		return true;	
	};

	// Release the thread.
	virtual void threadRelease() {

		publisher_.close();
		pd_->close();
		delete pd_;
	};

private:

	// Local and remote port.
	std::string local_;
	std::string remote_;

	// Topic to publish to.
	std::string topic_;

	// Publisher.
	yarp::os::Publisher<yarp::rosmsg::sensor_msgs::Image> publisher_;

	// Message counter.
	std::uint32_t msg_count_ = 0;

	// Device driver and options.
	yarp::dev::PolyDriver* pd_;

	yarp::os::Property options_;

	// Interface to device driver.
	yarp::dev::IFrameGrabberImage* fg_;

	// Image.
	yarp::sig::ImageOf<yarp::sig::PixelRgb> rgb_;
	yarp::sig::ImageOf<yarp::sig::PixelMono> mono_;
};


// YarpToRosImu implements a yarp::os::RateThread that
// reads data from a remote inertial measurement unit
// and publishes it to a Ros topic.
class YarpToRosImu : public yarp::os::RateThread
{
public:

	YarpToRosImu(int period, std::string local, std::string remote, std::string topic) : yarp::os::RateThread(period), local_(local), remote_(remote), topic_(topic), covariance_(9, 0) {	};

	// Call run every period milliseconds.
	virtual void run() {

		// Get inertial measurement.
		imu_ = imup_.LastRead();

		if (imu_.data() == YARP_NULLPTR) {

			return;
		}

		// Publish to ROS.		
		yarp::rosmsg::sensor_msgs::Imu& msg = publisher_.prepare();

		// Header.
                msg.header.seq = msg_count_++;
                msg.header.stamp = yarp::os::Time::now();
                msg.header.frame_id = "chassis";

		// Orientation.
                double euler_xyz[3], quaternion[4];

                euler_xyz[0] = imu_(0);
                euler_xyz[1] = imu_(1);
                euler_xyz[2] = imu_(2);

                convertEulerAngleYXZdegrees_to_quaternion(euler_xyz, quaternion);

                msg.orientation.x = quaternion[0];
                msg.orientation.y = quaternion[1];
                msg.orientation.z = quaternion[2];
                msg.orientation.w = quaternion[3];
                msg.orientation_covariance = covariance_;

		// Linear acceleration.
                msg.linear_acceleration.x = imu_(3);   // [m/s^2]
                msg.linear_acceleration.y = imu_(4);   // [m/s^2]
                msg.linear_acceleration.z = imu_(5);   // [m/s^2]
                msg.linear_acceleration_covariance = covariance_;

		// Angular acceleration.
                msg.angular_velocity.x = imu_(6);   // to be converted into rad/s (?) - verify with users
                msg.angular_velocity.y = imu_(7);   // to be converted into rad/s (?) - verify with users
                msg.angular_velocity.z = imu_(8);   // to be converted into rad/s (?) - verify with users
                msg.angular_velocity_covariance = covariance_;

		publisher_.write();
	};

	// Initialize the thread. On false return, thread doesn't call run and exits.
	virtual bool threadInit() {

		// Publisher.
		if (!publisher_.topic(topic_)) {

			std::cerr << "Could not publish to topic " << topic_ << std::endl;
			std::exit(1);
		};

		// Port to device.
		if (!imup_.open(local_)) {

			std::cerr << "Could not open port " << local_ << std::endl;
			std::exit(1);
		};

		// Connect to remote device.
		if (!yarp::os::Network::connect(remote_, local_)) {

			std::cerr << "Could not connect remote " << remote_ << " with local " << local_ << std::endl;
			std::exit(1);
		}
	};

	// Release the thread.
	virtual void threadRelease() {

		publisher_.close();
		imup_.close();
	};

private:

	// Local and remote port.
	std::string local_;
	std::string remote_;

	// Topic to publish to.
	std::string topic_;

	// Publisher.
	yarp::os::Publisher<yarp::rosmsg::sensor_msgs::Imu> publisher_;

	// Zero matrix to store covariance needed by Ros msg.
	std::vector<yarp::os::NetFloat64> covariance_;

	// Message counter.
	std::uint32_t msg_count_ = 0;

	// Port to device.
	ImuPort imup_;

	// Imu.
	yarp::sig::VectorOf<yarp::os::NetFloat64> imu_;
};


// Forward declare threads.
YarpToRosImg* l_cam;
YarpToRosImg* r_cam;
YarpToRosImu* imu;


// This is called when Ctrl-C is pressed. It stops
// all threads and then exits.
void my_handler(int s) {

	printf("Caught signal %d\n", s);
	
	l_cam->stop();
	r_cam->stop();
	imu->stop();
	
	delete l_cam;
	delete r_cam;
	delete imu;

	std::exit(EXIT_SUCCESS); 
}


// Main.
int main(int argc, char** argv)
{
	// Initialize network.
	yarp::os::Network yarp;
	yarp::os::Node node("/yarp_to_ros");

	// Call my_handler() when a SIGINT signal is received.
	struct sigaction sigIntHandler;
	sigIntHandler.sa_handler = my_handler;
	sigemptyset(&sigIntHandler.sa_mask);
	sigIntHandler.sa_flags = 0;
	sigaction(SIGINT, &sigIntHandler, NULL);

	// Initialize threads.
	l_cam = new YarpToRosImg(10, "/client/cam/left", "/irobot/cam/left", "/cam0/image_raw");
	r_cam = new YarpToRosImg(10, "/client/cam/right", "/irobot/cam/right", "/cam1/image_raw");
	imu = new YarpToRosImu(10, "/client/inertial", "/irobot/inertial", "/imu0");

	l_cam->start();
	r_cam->start();
	imu->start();

	// Pause the main thread until an interrupt is received.
	system("read -p '\nPress enter to continue or CTRL-C to abort...\n\n' var");
	
	l_cam->stop();
	r_cam->stop();
	imu->stop();

	delete l_cam;
	delete r_cam;
	delete imu;

	return 0;
}

