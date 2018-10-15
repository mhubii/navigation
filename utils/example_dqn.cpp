#include <torch/torch.h>

#include "models.h"

struct Options {

    std::string data_root{"data"};
    int32_t batch_size{64};
    int32_t epochs{10};
    double lr{0.01};
    double momentum{0.5};
    bool no_cuda{false};
    int32_t seed{1};
    int32_t test_batch_size{1000};
    int32_t log_interval{10};
};

int main(int argc, char** argv) {

    Options options;
    torch::DeviceType device_type;

    if (torch::cuda::is_available() && !options.no_cuda) {

        std::cout << "CUDA available! Training on GPU" << std::endl;
        device_type = torch::kCUDA;
    } 
    else {

        std::cout << "Training on CPU" << std::endl;
        device_type = torch::kCPU;
    }

    torch::Device device(device_type);

    DQN model;

    model.to(device);

    torch::Tensor in = torch::ones({1, 3, 64, 64});

    torch::Tensor out = model.forward(in);

    return 0;
}