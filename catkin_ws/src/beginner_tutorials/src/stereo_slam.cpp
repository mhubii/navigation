#include <vector>
#include <thread>
#include <signal.h>

#include <System.h>

#include <yarp/os/all.h>
#include <yarp/sig/all.h>
#include <yarp/dev/all.h>

#include <opencv2/opencv.hpp>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl/common/transforms.h>
#include <octomap/octomap.h>
#include <octomap/ColorOcTree.h>

typedef float msgtype;

// ORBSlam impelemts a yarp::os::RateThread that
// reads stereo camera images and performs simultanious
// locatization and mapping on them.
class ORBSlam : public yarp::os::RateThread
{
public:

	ORBSlam(int period, std::string vocab, std::string config, std::vector<std::string> locals, std::vector<std::string> remotes) : yarp::os::RateThread(period), locals_(locals), remotes_(remotes), pds_(locals.size()), fgs_(locals.size()), imgs_(locals.size()), imgs_cv_(locals.size()), vocab_(vocab), config_(config) {

		// Pre-check.
		if (locals_.size() != remotes_.size()) {
		
			std::cerr << "Didn't receive same number of local ports, remote ports and topics." << std::endl;
			std::exit(1);	
		}

		msg_count_ = 0;
	
		// OctoMap.
		tree_ = boost::make_shared<octomap::ColorOcTree>(0.1);
	};

	virtual void run() {

		// Read images and convert them.
		for (int i = 0; i < locals_.size(); i++) {
		
			fgs_[i]->getImage(imgs_[i]);

			// Convert the images to a format that OpenCV uses.
			imgs_cv_[i] = cv::cvarrToMat(imgs_[i].getIplImage());

			// Convert to gray image.
			// cv::cvtColor(img_cv_[camera], img_cv_[camera], cv::COLOR_BGR2GRAY);
		}

		// Track stereo.
		slam_->TrackStereo(imgs_cv_[0], imgs_cv_[1], yarp::os::Time::now());

		std::vector<ORB_SLAM2::MapPoint*> mp = slam_->GetTrackedMapPoints();

		if (mp.empty()) {

			return;
		}

		const int size = mp.size();

 		for (int i = 0; i < size; i++) {

			if (mp[i] == NULL) {

				continue;
			}

			cv::Mat mp_cv = mp[i]->GetWorldPos();
			//std::cout << mp.size() << std::endl;

			// xyz.
			float x = mp_cv.at<msgtype>(0);
			float y = mp_cv.at<msgtype>(1);
			float z = mp_cv.at<msgtype>(2);

			octomap::ColorOcTreeNode* col = tree_->updateNode( octomap::point3d(x, y, z), true );
		
/*
			if (col != NULL) {
			
				col->setColor(255, 0, 0);
				std::cout << "x: " << x << std::endl;
				std::cout << "y: " << y << std::endl;
				std::cout << "z: " << z << std::endl;
			}
*/

		}

	
		// Update OctoMap.
		tree_->updateInnerOccupancy();

/*	
		std::vector<ORB_SLAM2::MapPoint*> mp = slam_->GetTrackedMapPoints();

		if (mp.empty()) {

			return;
		}

		const int size = mp.size();
	
		// ROS.
		sensor_msgs::PointCloud2 msg;

		msg.header.frame_id = "sensor";
		msg.header.stamp = ros::Time::now();
		msg.header.seq = msg_count_++;

		msg.fields.resize(3);
		msg.fields[0].name = "x";
		msg.fields[0].offset = 0*sizeof(msgtype);
		msg.fields[0].datatype = 7;
		msg.fields[0].count = 1;

		msg.fields[1].name = "y";
		msg.fields[1].offset = 1*sizeof(msgtype);
		msg.fields[1].datatype = 7;
		msg.fields[1].count = 1;

		msg.fields[2].name = "z";
		msg.fields[2].offset = 2*sizeof(msgtype);
		msg.fields[2].datatype = 7;
		msg.fields[2].count = 1;

		msg.height = 1;
		msg.width = mp.size();

		msg.point_step = 3*sizeof(msgtype);
		msg.row_step = msg.width*msg.point_step;

		msg.is_dense = distance_threshold_;
		msg.is_bigendian = false;

		msg.data.resize(msg.point_step*msg.width);

 		for (int i = 0; i < size; i++) {

			if (mp[i] == NULL) {

				continue;
			}

			cv::Mat mp_cv = mp[i]->GetWorldPos();
			//std::cout << mp.size() << std::endl;

			// xyz.
			msgtype x = mp_cv.at<msgtype>(0);
			msgtype y = mp_cv.at<msgtype>(1);
			msgtype z = mp_cv.at<msgtype>(2);

			//std::cout << x << std::endl;

			*(reinterpret_cast<msgtype*>(&msg.data[0] + i*msg.point_step + 0*sizeof(msgtype))) = x;
			*(reinterpret_cast<msgtype*>(&msg.data[0] + i*msg.point_step + 1*sizeof(msgtype))) = y;
			*(reinterpret_cast<msgtype*>(&msg.data[0] + i*msg.point_step + 2*sizeof(msgtype))) = z;
		}
				
		pub_.publish(msg);
*/
	};

	virtual bool threadInit() {

		for (int i = 0; i < locals_.size(); i++) {
			
			// Device driver.
			yarp::os::Property options;

			options.put("device", "remote_grabber");
			options.put("local", locals_[i]);
			options.put("remote", remotes_[i]);

			pds_[i] = new yarp::dev::PolyDriver(options);

			if (!pds_[i]->isValid()) {

				std::cerr << "Device driver remote_grabber not available." << std::endl;
				return false;
			}

			if (!pds_[i]->view(fgs_[i])) {

				std::cerr << "Coult not acquire interface." << std::endl;
				return false;
			}
		}

		// Initialize system threads and get ready to process frames.
		slam_ = new ORB_SLAM2::System(vocab_.c_str(), config_.c_str(), ORB_SLAM2::System::STEREO, true);

		// ROS.
		nh_ = new ros::NodeHandle("rgbdslam");
		nh_->param<msgtype>("DistanceThreshold", distance_threshold_, 0.0);

		pub_ = nh_->advertise<sensor_msgs::PointCloud2>("batch_clouds", 100);

		return true;
	};

	virtual void threadRelease() {

	
		// Save OctoMap
		tree_->write( "yellow.ot" );
		cout<<"save octomap ... done."<<endl;

		for (int i = 0; i < locals_.size(); i++) {
			
			pds_[i]->close();
			delete pds_[i];
		}

		// Stop all threads.
		slam_->Shutdown();

		delete slam_;

		// ROS.
		delete nh_;
	};


private:

	// Local and remote port.
	std::vector<std::string> locals_;
	std::vector<std::string> remotes_;
	
	// Device driver and options.
	std::vector<yarp::dev::PolyDriver*> pds_;

	// Interface to device driver.
	std::vector<yarp::dev::IFrameGrabberImage*> fgs_;

	// Image.
	std::vector<yarp::sig::ImageOf<yarp::sig::PixelRgb>> imgs_;
	std::vector<cv::Mat> imgs_cv_;

	// SLAM system.
	ORB_SLAM2::System* slam_;

	// Vocabulary and configs.
	std::string vocab_;
	std::string config_;

	// ROS.
	ros::NodeHandle* nh_;
	ros::Publisher pub_;
	msgtype distance_threshold_;

	std::uint32_t msg_count_;

	// OctoMap.
	boost::shared_ptr<octomap::ColorOcTree> tree_;
};

ORBSlam* slam;

// This is called when Ctrl-C is pressed. It stops
// all threads and then exits.
void my_handler(int s) {

	printf("Caught signal %d\n", s);
	
	slam->stop();
	
	delete slam;

	std::exit(EXIT_SUCCESS); 
}

int main(int argc, char** argv)
{
	// Initialize network.
	yarp::os::Network yarp;
	ros::init(argc, argv, "rgbdslam");

	// Call my_handler() when a SIGINT signal is received.
	struct sigaction sigIntHandler;
	sigIntHandler.sa_handler = my_handler;
	sigemptyset(&sigIntHandler.sa_mask);
	sigIntHandler.sa_flags = 0;
	sigaction(SIGINT, &sigIntHandler, NULL);

	
	// Initialize the thread.
	int period = 10;
	std::string vocab = "/home/martin/Documents/path_finding/bin/config/ORBvoc.txt";
	std::string config = "/home/martin/Documents/path_finding/bin/config/heicub_stereo.yaml";
	std::vector<std::string> locals = {"/client/cam/left", "/client/cam/right"};
	std::vector<std::string> remotes = {"/vehicle/cam/left", "/vehicle/cam/right"};
	
	slam = new ORBSlam(period, vocab, config, locals, remotes);

	// Start the thread.
	slam->start();

	// Pause the main thread until an interrupt is received.
	system("read -p '\nPress enter to continue or CTRL-C to abort...\n\n' var");

	slam->stop();

	delete slam;	
/*
	std::string vocab = "/home/martin/Documents/path_finding/bin/config/ORBvoc.txt";
	std::string config = "/home/martin/Documents/path_finding/bin/config/heicub_stereo.yaml";
	std::vector<std::string> locals = {"/client/cam/left", "/client/cam/right"};
	std::vector<std::string> remotes = {"/vehicle/cam/left", "/vehicle/cam/right"};

	// Device driver and options.
	std::vector<yarp::dev::PolyDriver*> pds(2);

	// Interface to device driver.
	std::vector<yarp::dev::IFrameGrabberImage*> fgs(2);

	// Image.
	std::vector<yarp::sig::ImageOf<yarp::sig::PixelRgb>> imgs(2);
	std::vector<cv::Mat> imgs_cv(2);

	for (int i = 0; i < locals.size(); i++) {
	
		// Device driver.
		yarp::os::Property options;

		options.put("device", "remote_grabber");
		options.put("local", locals[i]);
		options.put("remote", remotes[i]);

		pds[i] = new yarp::dev::PolyDriver(options);
		pds[i]->view(fgs[i]);
	}


	ORB_SLAM2::System slam(vocab.c_str(), config.c_str(), ORB_SLAM2::System::STEREO, true);
	double now = yarp::os::Time::now();

	while(1) {

		// Read images and convert them.
		for (int i = 0; i < locals.size(); i++) {
		
			fgs[i]->getImage(imgs[i]);

			// Convert the images to a format that OpenCV uses.
			imgs_cv[i] = cv::cvarrToMat(imgs[i].getIplImage());
		}
		
		// Track stereo.
		slam.TrackStereo(imgs_cv[0], imgs_cv[1], yarp::os::Time::now());

		yarp::os::Time::delay(0.01);

		// std::vector<ORB_SLAM2::MapPoint*> mp(1, slam.GetTrackedMapPoints()[0]);


		// // const std::uint32_t size = mp.size();
		//if (!mp.empty()) {
		//	std::cout << mp.size() << std::endl;
		if (slam.GetTrackedMapPoints()[0] != NULL) {
			std::cout << slam.GetTrackedMapPoints()[0]->GetWorldPos() << std::endl;
		}
		//}

		if (yarp::os::Time::now() - now > 10) {

			break;
		}
	}

	slam.Shutdown();

	for (int i = 0; i < locals.size(); i++) {
		
		pds[i]->close();
		delete pds[i];
	}
*/
	return 0;
}
