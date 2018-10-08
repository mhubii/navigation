#include "read_cams.h"

ReadCams::ReadCams(std::vector<std::string> locals, std::vector<std::string> remotes) : locals_(locals), remotes_(remotes), pds_(locals.size()), fgs_(locals.size()), yarp_imgs_(locals.size()), cv_imgs_(locals.size()) {

	for (int i = 0; i < locals_.size(); i++) {
		
		// Device driver.
		yarp::os::Property options;

		options.put("device", "remote_grabber");
		options.put("local", locals_[i]);
		options.put("remote", remotes_[i]);

		pds_[i] = new yarp::dev::PolyDriver(options);

		if (!pds_[i]->isValid()) {

			std::cerr << "Device driver remote_grabber not available." << std::endl;
			std::exit(1);
		}

		if (!pds_[i]->view(fgs_[i])) {

			std::cerr << "Coult not acquire interface." << std::endl;
			std::exit(1);
		}
	}
};


ReadCams::~ReadCams() {

	for (int i = 0; i < locals_.size(); i++) {
		
		pds_[i]->close();
		delete pds_[i];
	}
};


std::vector<cv::Mat> ReadCams::GetCvImgs() {

	yarp_imgs_ = GetYarpImgs();

	for (int i = 0; i < locals_.size(); i++) {

		// Convert the images to a format that OpenCV uses.
		cv_imgs_[i] = cv::cvarrToMat(yarp_imgs_[i].getIplImage());

		cv::cvtColor(cv_imgs_[i], cv_imgs_[i], CV_RGB2BGR);
	}

	return cv_imgs_;
};

std::vector<yarp::sig::ImageOf<yarp::sig::PixelRgb>> ReadCams::GetYarpImgs() {

	for (int i = 0; i < locals_.size(); i++) {

		fgs_[i]->getImage(yarp_imgs_[i]);
	}

	return yarp_imgs_;
};

