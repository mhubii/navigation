#include <vector>
#include <signal.h>

#include <System.h>

#include <yarp/os/all.h>
#include <yarp/sig/all.h>
#include <yarp/dev/all.h>

#include <opencv2/opencv.hpp>


// ORBSlam impelemts a yarp::os::RateThread that
// reads stereo camera images and performs simultanious
// locatization and mapping on them.
class ORBSlam : public yarp::os::RateThread
{
public:

	ORBSlam(int period, std::string vocab, std::string config, std::vector<std::string> locals, std::vector<std::string> remotes) : yarp::os::RateThread(period), locals_(locals), remotes_(remotes), pds_(locals.size()), fgs_(locals.size()), imgs_(locals.size()), imgs_rgb_cv_(locals.size()), imgs_gray_cv_(locals.size()), vocab_(vocab), config_(config) {

		// Pre-check.
		if (locals_.size() != remotes_.size()) {
		
			std::cerr << "Didn't receive same number of local ports, remote ports and topics." << std::endl;
			std::exit(1);	
		}
	};

	virtual void run() {

		// Read images and convert them.
		for (int i = 0; i < locals_.size(); i++) {
		
			fgs_[i]->getImage(imgs_[i]);

			// Convert the images to a format that OpenCV uses.
			imgs_rgb_cv_[i] = cv::cvarrToMat(imgs_[i].getIplImage());

			// Convert to gray image.
			cv::cvtColor(imgs_rgb_cv_[i], imgs_gray_cv_[i], cv::COLOR_BGR2GRAY);
		}

		// Determine disparity.
		//l_matcher_->compute(imgs_gray_cv_[0], imgs_gray_cv_[1], l_disp_);
		//r_matcher_->compute(imgs_gray_cv_[1], imgs_gray_cv_[0], r_disp_);

		// Perform weighted least squares filtering.
		//wls_->filter(l_disp_, imgs_gray_cv_[0], wls_disp_, r_disp_);

		//cv::ximgproc::getDisparityVis(wls_disp_, wls_disp_, 1);
		// cv::normalize(wls_disp_, wls_disp_, 0, 255, CV_MINMAX, CV_8U);

		//cv::namedWindow("Depth View", cv::WINDOW_AUTOSIZE);
		//cv::imshow("Depth View", wls_disp_);
		//cv::waitKey(10);

		// Track stereo.
		slam_->TrackStereo(imgs_rgb_cv_[0], imgs_rgb_cv_[1], yarp::os::Time::now());

                // slam_->TrackRGBD(imgs_rgb_cv_[0], wls_disp_, yarp::os::Time::now());
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
		// slam_ = new ORB_SLAM2::System(vocab_.c_str(), config_.c_str(), ORB_SLAM2::System::RGBD, true);

		// Stereo matching and weighted least square filter.
		//l_matcher_ = cv::StereoBM::create(16, 9);
		//r_matcher_ = cv::ximgproc::createRightMatcher(l_matcher_);
		//wls_ = cv::ximgproc::createDisparityWLSFilter(l_matcher_);

		//wls_->setLambda(1e3);
		//wls_->setSigmaColor(1.5);

		return true;
	};

	virtual void threadRelease() {

		for (int i = 0; i < locals_.size(); i++) {
			
			pds_[i]->close();
			delete pds_[i];
		}

		// Stop all threads.
		slam_->Shutdown();

		std::vector<ORB_SLAM2::MapPoint*> points = slam_->GetTrackedMapPoints();

		if (points.size() != 0) {
			for (int i = 0; i < points.size(); i++) {

				if (points[i]->isBad())
					continue;
				//cv::Mat point = points[i]->GetWorldPos();
				//std::cout << point << std::endl;
			}
		}


		delete slam_;
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
	std::vector<cv::Mat> imgs_rgb_cv_;
	std::vector<cv::Mat> imgs_gray_cv_;

        // Stereo matching and weighted least square filter.
        //cv::Ptr<cv::StereoBM> l_matcher_;
        //cv::Ptr<cv::StereoMatcher> r_matcher_;
        //cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_;

        // Disparity map.
        //cv::Mat l_disp_; 
        //cv::Mat r_disp_; 
        //cv::Mat wls_disp_;

	// SLAM system.
	ORB_SLAM2::System* slam_;

	// Vocabulary and configs.
	std::string vocab_;
	std::string config_;
};

/*
typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;

pcl::PointCloud<PointT>::Ptr GeneratePointCloud(cv::Mat& color, cv::Mat& depth) {

	float fx = 475.739765;
	float fy = 475.739765;
	float cx = 285.090210;
	float cy = 256.762772;

	PointCloud::Ptr cloud(new PointCloud);

	for (int m=0; m < depth.rows; m++) {
		for (int n=0; n < depth.cols; n++) {
			
			float d = depth.at<float>(m, n);

			PointT p;
			p.z = d;
			p.x = ( n - cx) * p.z / fx;
			p.y = ( m - cy) * p.z / fy;

			p.r = color.at<cv::Vec<unsigned char, 3>>(m, n)[0];
			p.g = color.at<cv::Vec<unsigned char, 3>>(m, n)[1];
			p.b = color.at<cv::Vec<unsigned char, 3>>(m, n)[2];

			cloud->points.push_back(p);
		}
	}

	return cloud;
}



int main(int argc, char** argv)
{
	// Initialize yarp.
	yarp::os::Network yarp;

	std::vector<std::string> locals = {"/client/cam/left", "/client/cam/right"};
	std::vector<std::string> remotes = {"/vehicle/cam/left", "/vehicle/cam/right"};

	// Device driver and options.
	std::vector<yarp::dev::PolyDriver*> pds(locals.size());

	// Interface to device driver.
	std::vector<yarp::dev::IFrameGrabberImage*> fgs(locals.size());

	for (int i = 0; i < locals.size(); i++) {
		
		// Device driver.
		yarp::os::Property options;

		options.put("device", "remote_grabber");
		options.put("local", locals[i]);
		options.put("remote", remotes[i]);

		pds[i] = new yarp::dev::PolyDriver(options);

		// Interface.
		pds[i]->view(fgs[i]);
	}

	// Read images.
	std::vector<yarp::sig::ImageOf<yarp::sig::PixelRgb>> imgs(locals.size());
	std::vector<cv::Mat> imgs_rgb_cv(locals.size());
	std::vector<cv::Mat> imgs_gray_cv(locals.size());

	for (int i = 0; i < locals.size(); i++) {
	
		fgs[i]->getImage(imgs[i]);

		// Convert the images to a format that OpenCV uses.
		imgs_rgb_cv[i] = cv::cvarrToMat(imgs[i].getIplImage());

		// Convert to gray image.
		cv::cvtColor(imgs_rgb_cv[i], imgs_gray_cv[i], cv::COLOR_BGR2GRAY);
	}

	// Determine disparity.
        cv::Ptr<cv::StereoBM> l_matcher = cv::StereoBM::create(16, 9);
        cv::Ptr<cv::StereoMatcher> r_matcher = cv::ximgproc::createRightMatcher(l_matcher);
        cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls = cv::ximgproc::createDisparityWLSFilter(l_matcher);

	wls->setLambda(1e3);
	wls->setSigmaColor(1.5);

        // Disparity map.
        cv::Mat l_disp; 
        cv::Mat r_disp; 
        cv::Mat wls_disp;

	l_matcher->compute(imgs_gray_cv[0], imgs_gray_cv[1], l_disp);
	r_matcher->compute(imgs_gray_cv[1], imgs_gray_cv[0], r_disp);

	// Perform weighted least squares filtering.
	wls->filter(l_disp, imgs_gray_cv[0], wls_disp, r_disp);
	cv::normalize(wls_disp, wls_disp, 0, 1, CV_MINMAX, CV_32F);

	// Show image.
        cv::namedWindow("Depth View", cv::WINDOW_AUTOSIZE);
	cv::imshow("Depth View", wls_disp);
	cv::waitKey();

	// Convert to depth.

	// Create point cloud.
	PointCloud::Ptr pc = GeneratePointCloud(imgs_rgb_cv[0], wls_disp );

	pcl::visualization::CloudViewer viewer("pc");
	viewer.showCloud(pc);

	while (!viewer.wasStopped ())
	{
	}

	cv::normalize(l_disp, l_disp, 0, 1, CV_MINMAX, CV_32F);

	PointCloud::Ptr pc1 = GeneratePointCloud(imgs_rgb_cv[0], l_disp );

	pcl::visualization::CloudViewer viewer1("pc");
	viewer1.showCloud(pc1);

	while (!viewer1.wasStopped ())
	{
	}



	// Cleanup.
	for (int i = 0; i < locals.size(); i++) {
		
		pds[i]->close();
		delete pds[i];
	}

	return 0;
}
*/


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

	// Call my_handler() when a SIGINT signal is received.
	struct sigaction sigIntHandler;
	sigIntHandler.sa_handler = my_handler;
	sigemptyset(&sigIntHandler.sa_mask);
	sigIntHandler.sa_flags = 0;
	sigaction(SIGINT, &sigIntHandler, NULL);

	
	// Initialize the thread.
	int period = 10;
	std::string vocab = "config/ORBvoc.txt";
	std::string config = "config/heicub_stereo.yaml";
	std::vector<std::string> locals = {"/client/cam/left", "/client/cam/right"};
	std::vector<std::string> remotes = {"/vehicle/cam/left", "/vehicle/cam/right"};
	
	slam = new ORBSlam(period, vocab, config, locals, remotes);

	// Start the thread.
	slam->start();

	// Pause the main thread until an interrupt is received.
	system("read -p '\nPress enter to continue or CTRL-C to abort...\n\n' var");

	slam->stop();

	delete slam;	

	return 0;
}
