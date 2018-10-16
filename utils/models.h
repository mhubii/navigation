#ifndef MODELS_H_
#define MODELS_H_ 

#include <torch/torch.h>

struct DQN : torch::nn::Module {

    // Constructor.
    DQN()
        : conv1_(torch::nn::Conv2dOptions(3, 16, 5).stride(2)),
          bn1_(16),
          conv2_(torch::nn::Conv2dOptions(16, 32, 5).stride(2)),
          bn2_(32),
          conv3_(torch::nn::Conv2dOptions(32, 32, 5).stride(2)),
          bn3_(32),
          fc1_(448, 6) {    }

    // Forward pass.
    torch::Tensor forward(torch::Tensor x) {

        // Convolutional layers.
        x = torch::relu(bn1_->forward(conv1_->forward(x)));
        x = torch::relu(bn2_->forward(conv2_->forward(x)));
        x = torch::relu(bn3_->forward(conv3_->forward(x)));

        // Fully connected layer.
        x = fc1_->forward(torch::flatten(x));

        return x;
    }

    torch::nn::Conv2d conv1_;
    torch::nn::BatchNorm bn1_;
    torch::nn::Conv2d conv2_;
    torch::nn::BatchNorm bn2_;
    torch::nn::Conv2d conv3_;
    torch::nn::BatchNorm bn3_;
    torch::nn::Linear fc1_;
};

struct DRQN : torch::nn::Module {

    // TODO
};

#endif // MODELS_H_