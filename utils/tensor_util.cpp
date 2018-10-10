#include "tensor_util.h"

namespace caffe2 {

TensorUtil::TensorUtil(Tensor<CPUContext>& tensor) : tensor_(tensor) {

}

void TensorUtil::WriteImages(const std::string& name, float mean, bool lossy, int index) {

	int count = tensor_.dim(0);

	for (int i = 0; i < count; i++) {
	
		std::string suffix = index >= 0 ? "_" + std::to_string(i + index) : "";
		WriteImage(name + suffix, i, mean, lossy);
	}
}

void TensorUtil::WriteImage(const std::string& name, float mean, bool lossy) {

	cv::Mat image = TensorToImage(tensor_, index, 1.0, mean);
	std::string filename = name + (lossy ? ".jpg" : ".png");

	vector<int> params({CV_IMWRITE_JPEG_QUALITY, 90});

	CAFFE_ENFORCE(cv::imwrite(filename, image, params), "Unable to write to " + filename);
}

void TensorUtil::ReadImages(const std::vector<std::string>& filenames, int width, int height, std::vector<int>& indices, float mean, TensorProto::DataType type) {

}

void TensorUtil:: ReadImage(const std::string& filename, int width, int height) {

}
} // End of namespace caffe2.
