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

	ORBSlam(int period, std::string vocab, std::string config, std::vector<std::string> locals, std::vector<std::string> remotes) : yarp::os::RateThread(period), locals_(locals), remotes_(remotes), pds_(locals.size()), fgs_(locals.size()), imgs_(locals.size()), imgs_cv_(locals.size()), vocab_(vocab), config_(config) {

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
			imgs_cv_[i] = cv::cvarrToMat(imgs_[i].getIplImage());

			// Convert to gray image.
			// cv::cvtColor(img_cv_[camera], img_cv_[camera], cv::COLOR_BGR2GRAY);
		}

		// Track stereo.
		slam_->TrackStereo(imgs_cv_[0], imgs_cv_[1], yarp::os::Time::now());
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

		return true;
	};

	virtual void threadRelease() {

		for (int i = 0; i < locals_.size(); i++) {
			
			pds_[i]->close();
			delete pds_[i];
		}

		delete slam_;

		// Stop all threads.
		slam_->Shutdown();
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
	std::vector<std::string> remotes = {"/irobot/cam/left", "/irobot/cam/right"};
	
	slam = new ORBSlam(period, vocab, config, locals, remotes);

	// Start the thread.
	slam->start();

	// Pause the main thread until an interrupt is received.
	system("read -p '\nPress enter to continue or CTRL-C to abort...\n\n' var");

	slam->stop();

	delete slam;	
	
	return 0;
}
