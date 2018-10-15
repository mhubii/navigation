#include "blob_util.h"

#ifdef WITH_CUDA
    #include <caffe2/core/context_gpu.h>
#endif

namespace caffe2 {


BlobUtil::BlobUtil(Blob& blob) 
    : blob_(blob) { };

Tensor BlobUtil::Get() {

    #ifdef WITH_CUDA
        
        if (blob_.IsType<TensorCUDA>()) {
        
            return Tensor(blob_.Get<TensorCUDA>());
        }
    #endif

    return blob_.Get<Tensor>();
}

void BlobUtil::Set(const Tensor &value, bool force_cuda) {

    #ifdef WITH_CUDA
    
        if (force_cuda || blob_.IsType<TensorCUDA>()) {
    
            auto tensor = blob_.GetMutable<TensorCUDA>();
            tensor->CopyFrom(value);
            return;
        }
    #endif
    
    Tensor* tensor = blob_.GetMutable<Tensor>();
    tensor->ResizeLike(value);
    tensor->ShareData(value);
}

void BlobUtil::Print(const std::string &name, int max) {

    Tensor tensor = Get();
    TensorUtil(tensor).Print(name, max);
}
} // namespace caffe2