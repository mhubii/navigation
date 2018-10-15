#ifndef BLOB_UTIL_H_
#define BLOB_UTIL_H_

#include <caffe2/core/blob.h>
#include <caffe2/core/tensor.h>

#include "tensor_util.h"

namespace caffe2 {

class BlobUtil {

public:

    BlobUtil(Blob &blob);

    Tensor Get();

    void Set(const Tensor& value, bool force_cuda = false);
    void Print(const std::string &name = "", int max = 100);

protected:

    Blob& blob_;
};

}  // caffe2

#endif // UTIL_BLOB_H_