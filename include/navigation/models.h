#ifndef MODELS_H_
#define MODELS_H_ 

#include <torch/torch.h>

class Model : public torch::nn::Module {

    public:

        Model() {   };

        ~Model() {  };

        virtual torch::Tensor forward(torch::Tensor x) = 0;
};

// Implements a Deep Q-Network.
class DQN : public Model {

    public:

        // Constructor.
        DQN()
            : conv1_(torch::nn::Conv2dOptions(3, 16, 5).stride(2)),
              bn1_(16),
              conv2_(torch::nn::Conv2dOptions(16, 32, 5).stride(2)),
              bn2_(32),
              conv3_(torch::nn::Conv2dOptions(32, 32, 5).stride(2)),
              bn3_(32),
              fc1_(448, 6) {

            register_module("conv1", conv1_);
            register_module("bn1", bn1_);
            register_module("conv2", conv2_);
            register_module("bn2", bn2_);
            register_module("conv3", conv3_);
            register_module("bn3", bn3_);
            register_module("fc1", fc1_);
        }

        // Forward pass.
        torch::Tensor forward(torch::Tensor x) {

            // Convolutional layers.
            x = torch::relu(bn1_->forward(conv1_->forward(x)));
            x = torch::relu(bn2_->forward(conv2_->forward(x)));
            x = torch::relu(bn3_->forward(conv3_->forward(x)));

            // Fully connected layer.
            //x = fc1_->forward(torch::flatten(x));

            return x;
        }

    private:

        torch::nn::Conv2d conv1_;
        torch::nn::BatchNorm bn1_;
        torch::nn::Conv2d conv2_;
        torch::nn::BatchNorm bn2_;
        torch::nn::Conv2d conv3_;
        torch::nn::BatchNorm bn3_;
        torch::nn::Linear fc1_;
};

// Implements a Deep Recurrent Q-Network.
class DRQN : public Model {

    public:
    
        // TODO

    private:

        // TODO
};

#endif // MODELS_H_
