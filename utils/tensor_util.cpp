#include "tensor_util.h"

namespace caffe2 {

TensorUtil::TensorUtil(Tensor& tensor) : tensor_(tensor) {

}

template <typename T> 
void image_to_tensor(Tensor& tensor, cv::Mat& image, float mean) {

	// Split image into channels.
	image.convertTo(image, CV_32FC3, 1.0, -mean);
	vector<cv::Mat> channels(3);
	cv::split(image, channels);

	// Convert cv::Mat to std::vector<T>.
	std::vector<T> data;

	for (auto& c : channels) {

		data.insert(data.end(), (T*)c.datastart, (T*)c.dataend);
	}
	
	// Fill tensor with data.
	std::vector<TIndex> dims({1, 3, image.rows, image.cols});
	Tensor t(dims, data, NULL);
	tensor.ResizeLike(t);
	tensor.ShareData(t);
}

template <typename T> 
cv::Mat tensor_to_image(const Tensor& tensor, int index, float scale, float mean, int type) {

	// Assure tensor has dimensionality 4.
	CAFFE_ENFORCE_EQ(tensor.ndim(), 4);
	
	int count = tensor.dim(0);
	int depth = tensor.dim(1);
	int height = tensor.dim(2);
	int width = tensor.dim(3);

	CAFFE_ENFORCE_LT(index, count);

	// Convert tensor to cv::Mat and iterate over channels.
	const T* data = tensor.data<T>() + (index*width*height*depth); 
	
	std::vector<cv::Mat> channels(depth);

	for (auto& j : channels) {

		j = cv::Mat(height, width, type, (void*)data);
		
		if (scale != 1.0 || mean != 0.0) {

			cv::Mat k;
			j.convertTo(k, type, scale, mean);
			j = k;
		}

		data += (width*height);
	}	

	// Merge channels.
	cv::Mat image;
	cv::merge(channels, image);

	if (depth == 1) {

		cv::cvtColor(image, image, CV_GRAY2RGB);
	}

	return image;
}

cv::Mat tensor_to_image(const Tensor& tensor, int index, float scale, float mean) {

	if (tensor.IsType<float>()) {

		return tensor_to_image<float>(tensor, index, scale, mean, CV_32F);
	}
	if (tensor.IsType<uchar>()) {

		return tensor_to_image<uchar>(tensor, index, scale, mean, CV_8UC1);
	}

	LOG(FATAL) << "tensor_to_image() for type " << tensor.meta().name()
		   << " not implemented.";
}

template <typename T>
void TensorUtil::ReadImageTensor(Tensor& tensor, const std::vector<std::string>& filenames, int width, int height, std::vector<int>& indices, float mean, TensorProto::DataType type) {

	std::vector<T> data;
	data.reserve(filenames.size() * 3 * width * height);
	int count = 0;

	for (auto &filename : filenames) {

		// Load image.
		cv::Mat image = cv::imread(filename);  // CV_8UC3 uchar

		if (!image.cols || !image.rows) {
			
			count++;
			continue;
		}

		if (image.cols != width || image.rows != height) {
			
			// Scale image to fit.
			cv::Size scaled(std::max(height * image.cols / image.rows, width), std::max(height, width * image.rows / image.cols));
			cv::resize(image, image, scaled);

			// Crop image to fit.
			cv::Rect crop((image.cols - width) / 2, (image.rows - height) / 2, width, height);
			image = image(crop);
		}

		switch (type) {
			
			case TensorProto_DataType_FLOAT:
				image.convertTo(image, CV_32FC3, 1.0, -mean);
				break;
			case TensorProto_DataType_INT8:
				image.convertTo(image, CV_8SC3, 1.0, -mean);
				break;
			default:
				break;
		}


		CAFFE_ENFORCE_EQ(image.channels(), 3);
		CAFFE_ENFORCE_EQ(image.rows, height);
		CAFFE_ENFORCE_EQ(image.cols, width);

		// convert NHWC to NCHW
		vector<cv::Mat> channels(3);
		cv::split(image, channels);

		for (auto &c : channels) {

			data.insert(data.end(), (T *)c.datastart, (T *)c.dataend);
		}

		indices.push_back(count++);
	}

	// Create tensor.
	std::vector<TIndex> dims({(TIndex)indices.size(), 3, height, width});
	Tensor t(dims, data, NULL);
	tensor.ResizeLike(t);
	tensor.ShareData(t);
}

template <typename T>
void TensorUtil::TensorPrintType(const Tensor& tensor, const std::string &name,
								 int max) {

	const auto &data = tensor.template data<T>();
	
	if (name.length() > 0) std::cout << name << "(" << tensor.dims() << "): ";
	
		for (auto i = 0; i < (tensor.size() > max ? max : tensor.size()); ++i) {
	
			std::cout << (float)data[i] << ' ';
		}
	
	if (tensor.size() > max) {
		
		std::cout << "... (" << *std::min_element(data, data + tensor.size()) << ","
		    	  << *std::max_element(data, data + tensor.size()) << ")";
	}
	
	if (name.length() > 0) std::cout << std::endl;
}

void TensorUtil::WriteImages(const std::string& name, float mean, bool lossy, int index) {

	int count = tensor_.dim(0);

	for (int i = 0; i < count; i++) {
	
		std::string suffix = index >= 0 ? "_" + std::to_string(i + index) : "";
		WriteImage(name + suffix, i, mean, lossy);
	}
}

void TensorUtil::WriteImage(const std::string& name, int index, float mean, bool lossy) {

	cv::Mat image = tensor_to_image(tensor_, index, 1.0, mean);
	std::string filename = name + (lossy ? ".jpg" : ".png");

	vector<int> params({CV_IMWRITE_JPEG_QUALITY, 90});

	CAFFE_ENFORCE(cv::imwrite(filename, image, params), "Unable to write to " + filename);
}

void TensorUtil::ReadImages(const std::vector<std::string>& filenames, int width, int height, std::vector<int>& indices, float mean, TensorProto::DataType type) {

}

void TensorUtil:: ReadImage(const std::string& filename, int width, int height) {

}

void TensorUtil::Print(const std::string &name, int max) {
	
	if (tensor_.template IsType<float>()) {
	
		return TensorPrintType<float>(tensor_, name, max);
	}
	
	if (tensor_.template IsType<int>()) {
	
		return TensorPrintType<int>(tensor_, name, max);
	}
	
	if (tensor_.template IsType<uint8_t>()) {
	
		return TensorPrintType<uint8_t>(tensor_, name, max);
	}
	
	if (tensor_.template IsType<int8_t>()) {
	
		return TensorPrintType<int8_t>(tensor_, name, max);
	}
	
	std::cout << name << "?" << std::endl;
}
} // End of namespace caffe2.
