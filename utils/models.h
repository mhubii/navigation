#ifndef MODELS_H_
#define MODELS_H_ 

#include <torch/torch.h>

struct DQN : torch::nn::Module {

    DQN()
        : conv1_(torch::nn::Conv2dOptions(3, 16, /*kernel_size=*/5)),
          bn1_(16),
          conv2_(torch::nn::Conv2dOptions(16, 32, /*kernel_size=*/5)),
          bn2_(32),
          conv3_(torch::nn::Conv2dOptions(32, 32, /*kernel_size=*/5)),
          bn3_(32) {
            
        // Stride in Conv2dOptions constructor not supported yet.
        conv1_->options.stride(2);
        conv2_->options.stride(2);
        conv3_->options.stride(2);
    }

    torch::Tensor forward(torch::Tensor x) {

        x = torch::relu(bn1_->forward(conv1_->forward(x)));
        x = torch::relu(bn2_->forward(conv2_->forward(x)));
        x = torch::relu(bn3_->forward(conv3_->forward(x)));

        return x;
    }

    torch::nn::Conv2d conv1_;
    torch::nn::BatchNorm bn1_;
    torch::nn::Conv2d conv2_;
    torch::nn::BatchNorm bn2_;
    torch::nn::Conv2d conv3_;
    torch::nn::BatchNorm bn3_;
    // torch::nn::Linear fc1_;
};

struct DRQN : torch::nn::Module {

    // TODO
};

#endif // MODELS_H_