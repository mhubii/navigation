#include <opencv2/opencv.hpp>
#include "read_cams.h"

int main(int argc, char** argv) {

	yarp::os::Network yarp;

	std::vector<std::string> locals = {"/client/cam/left", "/client/cam/right"};
	std::vector<std::string> remotes = {"/irobot/cam/left", "/irobot/cam/right"};

	ReadCams read_cams(locals, remotes);

	std::vector<cv::Mat> imgs = read_cams.GetCvImgs();


	cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE ); // Create a window for display.
	cv::imshow( "Display window", imgs[0] );                // Show our image inside it.
	cv::waitKey(0); // Wait for a keystroke in the window

	return 0;
}
