#ifndef TENSOR_UTIL_H_
#define TENSOR_UTIL_H_

#include <stdint.h>

#include <caffe2/core/tensor.h>
#include <opencv2/opencv.hpp>

namespace caffe2 {

class TensorUtil {

public:

	TensorUtil(Tensor& tensor);

	void WriteImages(const std::string& name, float mean, bool lossy, int index = 0);

	void WriteImage(const std::string& name, int index, float mean, bool lossy);

	void ReadImages(const std::vector<std::string>& filenames, int width, int height, std::vector<int>& indices, float mean, TensorProto::DataType type);

	void ReadImage(const std::string& filename, int width, int height);	

	void Print(const std::string& name = "", int max = 100);

protected:

	Tensor& tensor_;

private:

	template <typename T>
	void ReadImageTensor(Tensor& tensor, const std::vector<std::string>& filenames, int width, int height, std::vector<int>& indices, float mean, TensorProto::DataType type);

	template <typename T>
	void TensorPrintType(const Tensor& tensor, const std::string &name, int max);
};
} // End of namespace caffe2.

#endif // TENSOR_UTIL_H_
