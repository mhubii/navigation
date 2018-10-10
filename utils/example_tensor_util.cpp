#include <caffe2/core/tensor.h>
#include <stdio.h>

#include "tensor_util.h"
#include "read_cams.h"

int main(int argc, char **argv) {

	// Create yarp network.
	yarp::os::Network yarp;

	// Ports.
	std::vector<std::string> locals = {"/client/cam/left", "/client/cam/right"};
	std::vector<std::string> remotes = {"/irobot/cam/left", "/irobot/cam/right"};

	ReadCams read_cams(locals, remotes);

	// Read images from ports.
	std::vector<cv::Mat> imgs = read_cams.GetCvImgs();

	// Convert images to tensor and other way round.
	//caffe2::TensorCPU tensor;
	//caffe2::TensorUtil::ImageToTensor<float>(tensor, imgs[0], 128);	

	//cv::Mat img = caffe2::TensorUtil::TensorToImage(tensor, 0, 1.0, 128);

	// Show img.
	//img.convertTo(img, CV_8UC3);

	//cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE ); // Create a window for display.
	//cv::imshow( "Display window", img);                // Show our image inside it.
	//cv::waitKey(0); // Wait for a keystroke in the window

	return 0;
}
