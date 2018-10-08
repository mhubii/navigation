#ifndef READ_CAMS_H_
#define READ_CAMS_H_

#include <vector>
#include <yarp/dev/all.h>
#include <yarp/sig/all.h>
#include <opencv2/opencv.hpp>

// ReadCams offers an interface to simply read images from
// yarp ports.
class ReadCams
{
public:

	ReadCams(std::vector<std::string> locals, std::vector<std::string> remotes);

	~ReadCams();

	std::vector<cv::Mat> GetCvImgs();

	std::vector<yarp::sig::ImageOf<yarp::sig::PixelRgb>> GetYarpImgs();

private:

	// Local and remote ports.
	std::vector<std::string> locals_;
	std::vector<std::string> remotes_;

	// Device driver and options.
	std::vector<yarp::dev::PolyDriver*> pds_;

	// Interface to device driver.
	std::vector<yarp::dev::IFrameGrabberImage*> fgs_;

	// Images.
	std::vector<yarp::sig::ImageOf<yarp::sig::PixelRgb>> yarp_imgs_;
	std::vector<cv::Mat> cv_imgs_;
};

#endif
