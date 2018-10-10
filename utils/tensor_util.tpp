template <typename T> 
void TensorUtil::ImageToTensor(TensorCPU& tensor, cv::Mat& image, float mean) {

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
	TensorCPU t(dims, data, NULL);
	tensor.ResizeLike(t);
	tensor.ShareData(t);
}

template <typename T> 
cv::Mat TensorUtil::TensorToImage(const Tensor<CPUContext>& tensor, int index, float scale, float mean, int type) {

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

cv::Mat TensorUtil::TensorToImage(const Tensor<CPUContext>& tensor, int index, float scale, float mean) {

	if (tensor.IsType<float>()) {

		return TensorToImage<float>(tensor, index, scale, mean, CV_32F);
	}
	if (tensor.IsType<uchar>()) {

		return TensorToImage<uchar>(tensor, index, scale, mean, CV_8UC1);
	}

	LOG(FATAL) << "TensorToImage() for type " << tensor.meta().name()
		   << " not implemented.";
}

template <typename T>
void TensorUtil::ReadImageTensor(TensorCPU& tensor, const std::vector<std::string>& filenames, int width, int height, std::vector<int>& indices, float mean, TensorProto::DataType type) {

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
	TensorCPU t(dims, data, NULL);
	tensor.ResizeLike(t);
	tensor.ShareData(t);
}

