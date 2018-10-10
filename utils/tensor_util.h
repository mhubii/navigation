#ifndef TENSOR_UTIL_H_
#define TENSOR_UTIL_H_

#include <stdint.h>

#include <caffe2/core/tensor.h>
#include <opencv2/opencv.hpp>

namespace caffe2 {

class TensorUtil {

public:

	TensorUtil(Tensor<CPUContext>& tensor);

	void WriteImages(const std::string& name, float mean, bool lossy, int index);

	void WriteImage(const std::string& name, float mean, bool lossy);

	void ReadImages(const std::vector<std::string>& filenames, int width, int height, std::vector<int>& indices, float mean, TensorProto::DataType type);

	void ReadImage(const std::string& filename, int width, int height);	

protected:

	Tensor<CPUContext>& tensor_;

private:
	template <typename T> 
	void ImageToTensor(TensorCPU& tensor, cv::Mat& image, float mean = 128.0);

	template <typename T> 
	cv::Mat TensorToImage(const Tensor<CPUContext>& tensor, int index, float scale, float mean, int type);

	cv::Mat TensorToImage(const Tensor<CPUContext>& tensor, int index, float scale = 1.0, float mean = 128.0);

	template <typename T>
	void ReadImageTensor(TensorCPU& tensor, const std::vector<std::string>& filenames, int width, int height, std::vector<int>& indices, float mean, TensorProto::DataType type);
};

#include "tensor_util.tpp"

} // End of namespace caffe2.

#endif // TENSOR_UTIL_H_
