#include <iostream>
#include <vector>
#include <stdio.h>

#include "ros/ros.h"
#include "image_transport/image_transport.h"
#include "sensor_msgs/image_encodings.h"
#include "cv_bridge/cv_bridge.h"

#include <yarp/os/all.h>
#include <yarp/dev/all.h>
#include <yarp/rosmsg/TickTime.h>
#include <yarp/rosmsg/sensor_msgs/Imu.h>
#include <yarp/rosmsg/sensor_msgs/Image.h>

#include <opencv2/opencv.hpp>

// A port that stores the value last received.
class IMUPort : public yarp::os::Publisher<yarp::rosmsg::sensor_msgs::Imu>
{
public:

	IMUPort() {

		//setCallbackLock(&mutex_);
		setReader(imu_);
	}

	yarp::sig::Vector LastRead() {

		//lockCallback();
		//yarp::sig::Vector tmp = last_;
		//unlockCallback();
		
		//return tmp;		
	}

private:

	yarp::rosmsg::sensor_msgs::Imu imu_;
	yarp::sig::Vector last_;
	yarp::os::Mutex mutex_;
};

class IMGPort : public yarp::os::Port
{
public:

	IMGPort() {

		setCallbackLock(&mutex_);
		setReader(last_);
	}

	yarp::os::Bottle LastRead() {

		lockCallback();
		yarp::os::Bottle tmp = last_;
		unlockCallback();

		return tmp;
	};

private:

	yarp::os::Bottle last_;
	yarp::os::Mutex mutex_;
};


int main(int argc, char** argv)
{

	yarp::os::Network yarp;

	double now = yarp::os::Time::now();

	// Publish camera view.
	//yarp::os::Property options;

	//options.put("device", "remote_grabber");
	//options.put("local", "/client/cam0/image_raw");
	//options.put("remote", "/irobot/cam/left");

	//yarp::dev::PolyDriver* dd = new yarp::dev::PolyDriver(options);

	//yarp::dev::IFrameGrabberImage* f;
	//dd->view(f);

	//yarp::sig::ImageOf<yarp::sig::PixelRgb> img;
	//cv::Mat img_cv_rgb;
	//cv::Mat img_cv_gray;

	// Show image.
	//cv::namedWindow("Display Window", cv::WINDOW_AUTOSIZE);

	// Read IMG.
	//IMGPort img;
	//img.open("/client/cam/left");
	//yarp.connect("/irobot/cam/left", "/client/cam/left");
	
	// ROS (YARP).
	//yarp::os::Node ros_node("/yarp_to_ros/node");   

	//yarp::os::Publisher<yarp::rosmsg::sensor_msgs::Image> ros_pub_port; 
	//ros_pub_port.topic("/yarp_to_ros/topic");
	//yarp::rosmsg::sensor_msgs::Image& ros_data = ros_pub_port.prepare();

	//while (yarp::os::Time::now() - now < 20) {
	
		//yarp::rosmsg::sensor_msgs::Image data;
		//std::cout << img.LastRead().toString() << std::endl;
		//data.writeBare(img.LastRead());

		//ros_pub_port.write(data);
	//}

	//img.close();



/*
	yarp::os::NetUint32 msg_counter = 0;

	while (yarp::os::Time::now() - now < 20) {

		f->getImage(img);

		// Convert the images to a format that OpenCV uses.
		img_cv_rgb = cv::cvarrToMat(img.getIplImage());

		// Convert to gray image.
		//cv::cvtColor(img_cv_rgb, img_cv_gray, cv::COLOR_RGB2GRAY);
		// Convert to unsigned int 8.
		//cv::normalize(img_cv, img_cv, 0, 255, CV_MINMAX, CV_8U);

		cv::imshow("Display Window", img_cv_rgb);

		cv::waitKey(10);
		yarp::os::Time::delay(0.01); // Publish an image as a ros::sensor_msgs::Image to a topic.

		// Write to ROS port.
		ros_data.header.seq = msg_counter++;
		ros_data.header.stamp = yarp::os::Time::now();
		ros_data.header.frame_id = "chassis";

		ros_data.height = std::uint32_t(img.height());
		ros_data.width = std::uint32_t(img.width());

		ros_data.encoding = "rgb8";
		ros_data.is_bigendian = std::uint8_t(0);
		ros_data.step = std::uint32_t(img.getRowSize());

		std::vector<std::uint8_t> data(img.getRawImage(), img.getRawImage() + img.getRawImageSize());

		ros_data.data = data;

		printf("Squence: %d\n", ros_data.header.seq);
		printf("Time  s: %d\n", ros_data.header.stamp.sec);
		printf("Time ns: %d\n", ros_data.header.stamp.nsec);
		printf("Height : %d\n", ros_data.height);
		printf("Width  : %d\n", ros_data.width);
		printf("Step   : %d\n", ros_data.step);
		printf("Pixel  : %d\n", img.getPixelSize());
		printf("Size   : %d\n", img.getRawImageSize());
		printf("Sample : %d\n", ros_data.data[0]);


		ros_pub_port.write();
		yarp::os::Time::delay(1);
	}
*/

	//ros_pub_port.interrupt();
	//ros_pub_port.close();

	//ros_node.interrupt();




/*
	// ROS.
	ros::init(argc, argv, "yarp_to_ros_pub");
	ros::NodeHandle nh;
	image_transport::ImageTransport it(nh);
	image_transport::Publisher pub = it.advertise("/cam0/image_raw", 1);

	while (yarp::os::Time::now() - now < 40) {

		f->getImage(img);

		// Convert the images to a format that OpenCV uses.
		img_cv_rgb = cv::cvarrToMat(img.getIplImage());

		// Convert to gray image.
		cv::cvtColor(img_cv_rgb, img_cv_gray, cv::COLOR_RGB2GRAY);

		// Opencv imshow.
		cv::imshow("Display Window", img_cv_rgb);

		cv::waitKey(10);
		yarp::os::Time::delay(0.01);

		// Publish on ros.
		sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "rgb8", img_cv_gray).toImageMsg();

		pub.publish(msg);
		ros::spinOnce();	
	}
*/	
	//dd->close();
	//delete dd;

	

	// Read IMU.
	yarp::os::Node node("/yarp/ros");

	//IMUPort imu;
	//imu.open("/client/intertial");
	//yarp.connect("/irobot/inertial", "/client/inertial");
	//imu.topic("/rostopic/inertial");


	// Extract images.
	yarp::os::Property options;

	options.put("device", "remote_grabber");
	options.put("local", "/client/cam0/image_raw");
	options.put("remote", "/irobot/cam/left");

	yarp::dev::PolyDriver* dd = new yarp::dev::PolyDriver(options);

	yarp::dev::IFrameGrabberImage* f;
	dd->view(f);

	yarp::sig::ImageOf<yarp::sig::PixelRgb> img;
	yarp::sig::ImageOf<yarp::sig::PixelMono> mono;


	// Publish to ROS.
	yarp::os::Publisher<yarp::rosmsg::sensor_msgs::Image> publisher;

	publisher.topic("/topic");

	std::uint32_t msg_count = 0;	

	while (yarp::os::Time::now() - now < 60) {

		// Get image.
		f->getImage(img);
		mono.copy(img);


		// Publish to ROS.		
		yarp::rosmsg::sensor_msgs::Image& msg = publisher.prepare();

		// Header
		msg.header.seq = msg_count++;
		msg.header.stamp = yarp::os::Time::now();
		msg.header.frame_id = "chassis";

		// Size.
		msg.height = mono.height();
		msg.width = mono.width();

		// Encoding.
		msg.encoding = "mono8";
		msg.is_bigendian = 0;
		msg.step = mono.getRowSize();

		// Data.
		//std::vector<std::uint8_t> data(10000, 2);
		//std::generate(data.begin(), data.end(),  [n = 0] () mutable { return n++%100; });

		std::vector<std::uint8_t> data(mono.getRawImage(), mono.getRawImage() + mono.getRawImageSize());
		msg.data = data;

		publisher.write();
	}

	//yarp::os::Time::delay(10);
	//yarp.connect("/irobot/inertial", "/client/inertial");
	
	//while (yarp::os::Time::now() - now < 10) {

	//	std::printf("IMU returns: %s\n", imu.LastRead().toString().c_str());	
	//}
	

	// Check for ServerInertial::run() to see how yarp publishes to ros.
	//yarp::os::Node node("/yarp/ros_pub");

	//yarp::os::Publisher<yarp::rosmsg::sensor_msgs::Imu> publisher;

	//if (!publisher.topic("/imu0")) {

	//	std::cerr << "Failed to create publisher to /imu0\n";
	//	return -1;
	//}

	//while (yarp::os::Time::now() - now < 20) {
	
	//	yarp::rosmsg::sensor_msgs::Imu data;

	//	data.writeBare(imu.LastRead());

	//	publisher.write(data);
	//}

	publisher.close();
	node.interrupt();
	dd->close();
	delete dd;

	// Publish on ROS topic.
	//ros::init(argc, argv, "yarp_to_ros_pub");

	//ros::NodeHandle n;

	//ros::Publisher pub = n.advertise<sensor_msgs::Imu>("imu0", 1000);

	return 0;
}
