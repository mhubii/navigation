#include <iostream>
#include <stdio.h>

#include "ros/ros.h"
#include "image_transport/image_transport.h"
#include "sensor_msgs/image_encodings.h"
#include "cv_bridge/cv_bridge.h"

#include <yarp/os/all.h>
#include <yarp/dev/all.h>
#include <yarp/rosmsg/sensor_msgs/Imu.h>
#include <yarp/rosmsg/sensor_msgs/Image.h>

#include <opencv2/opencv.hpp>

// A port that stores the value last received.
class IMUPort : public yarp::os::Port
{
public:

	IMUPort() {

		setCallbackLock(&mutex_);
		setReader(last_);
	}

	yarp::sig::Vector LastRead() {

		lockCallback();
		yarp::sig::Vector tmp = last_;
		unlockCallback();
		
		return tmp;		
	}

private:

	yarp::sig::Vector last_;
	yarp::os::Mutex mutex_;
};

int main(int argc, char** argv)
{

	yarp::os::Network yarp;

	double now = yarp::os::Time::now();

	// Publish camera view.
	yarp::os::Property options;

	options.put("device", "remote_grabber");
	options.put("local", "/client/cam0/image_raw");
	options.put("remote", "/irobot/cam/left");

	yarp::dev::PolyDriver* dd = new yarp::dev::PolyDriver(options);

	yarp::dev::IFrameGrabberImage* f;
	dd->view(f);

	yarp::sig::ImageOf<yarp::sig::PixelRgb> img;
	cv::Mat img_cv_rgb;
	cv::Mat img_cv_gray;

	// ROS (YARP).
	//yarp::os::Node* ros_node = new yarp::os::Node("/yarp_to_ros/node");   

	//yarp::os::Publisher<yarp::rosmsg::sensor_msgs::Image> ros_pub_port; 
	//ros_pub_port.topic("/yarp_to_ros/topic");
	//yarp::rosmsg::sensor_msgs::Image &ros_data = ros_pub_port.prepare();

	// ROS.
	ros::init(argc, argv, "yarp_to_ros_pub");
	ros::NodeHandle nh;
	image_transport::ImageTransport it(nh);
	image_transport::Publisher pub = it.advertise("/cam0/image_raw", 1);

	// Show image.
	cv::namedWindow("Display Window", cv::WINDOW_AUTOSIZE);

/*
	while (yarp::os::Time::now() - now < 40) {

		f->getImage(img);

		// Convert the images to a format that OpenCV uses.
		img_cv_rgb = cv::cvarrToMat(img.getIplImage());

		// Convert to gray image.
		cv::cvtColor(img_cv_rgb, img_cv_gray, cv::COLOR_BGR2GRAY);
		// Convert to unsigned int 8.
		//cv::normalize(img_cv, img_cv, 0, 255, CV_MINMAX, CV_8U);

		cv::imshow("Display Window", img_cv_gray);

		cv::waitKey(10);
		yarp::os::Time::delay(0.01); // Publish an image as a ros::sensor_msgs::Image to a topic.

		// Write to ROS port.
		//ros_data.height = std::uint32_t(img_cv_gray.rows);
		//ros_data.width = std::uint32_t(img_cv_gray.cols);

		//ros_data.encoding = sensor_msgs::image_encodings::RGB8;
		//ros_data.is_bigendian = std::uint8_t(0);
		//ros_data.step = std::uint32_t(sizeof(unsigned char)*img.width());

		//std::vector<std::uint8_t> data(img.getRawImage(), img.getRawImage()+img.width()*img.height());
		//ros_data.data = data;
		//ros_data = cv_bridge::CvImage(std_msgs::Header(), "rgb8", img_cv_rgb).toImageMsg();

		//ros_pub_port.write();

		sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "rgb8", img_cv).toImageMsg();
		ros::spinOnce();
		loop_rate.sleep();

	}
*/


	while (yarp::os::Time::now() - now < 40) {

		f->getImage(img);

		// Convert the images to a format that OpenCV uses.
		img_cv_rgb = cv::cvarrToMat(img.getIplImage());

		// Opencv imshow.
		cv::imshow("Display Window", img_cv_rgb);

		cv::waitKey(10);
		yarp::os::Time::delay(0.01);

		// Publish on ros.
		sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "rgb8", img_cv_rgb).toImageMsg();

		pub.publish(msg);
		ros::spinOnce();	
	}

	

	//ros_pub_port.interrupt();
	//ros_pub_port.close();

	//ros_node->interrupt();
	//delete ros_node;
	
	dd->close();
	delete dd;

	

	// Read IMU.
	//IMUPort imu;
	//imu.open("/client/inertial");
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

	// Publish on ROS topic.
	//ros::init(argc, argv, "yarp_to_ros_pub");

	//ros::NodeHandle n;

	//ros::Publisher pub = n.advertise<sensor_msgs::Imu>("imu0", 1000);
}
