#ifndef MODELS_H_
#define MODELS_H_ 

#include <torch/torch.h>

// Implements a convolutional neural net for stereo inputs. It maps states to actions.
class Actor : public torch::nn::Module {

    public:

        // Constructor.
        Actor(at::IntList input_shape /*{int64_t channels, int64_t height, int64_t width}*/, int64_t dof)
            : // Left layers.
              left_conv1_(torch::nn::Conv2dOptions(3, 8, 5).stride(2)),
              left_conv2_(torch::nn::Conv2dOptions(8, 16, 5).stride(2)),
              left_conv3_(torch::nn::Conv2dOptions(16, 32, 3).stride(2)),
            //   left_conv4_(torch::nn::Conv2dOptions(32, 64, 3).stride(2)),
            //   left_conv5_(torch::nn::Conv2dOptions(64, 64, 3).stride(2)),

              left_fc1_(GetConvOutput(input_shape), 16),
              left_fc2_(16, 8),
              left_fc3_(8, dof),
            //   left_fc4_(8, dof),

              // Right layers.
              right_conv1_(torch::nn::Conv2dOptions(3, 8, 5).stride(2)),
              right_conv2_(torch::nn::Conv2dOptions(8, 16, 5).stride(2)),
              right_conv3_(torch::nn::Conv2dOptions(16, 32, 3).stride(2)),
            //   right_conv4_(torch::nn::Conv2dOptions(32, 64, 3).stride(2)),
            //   right_conv5_(torch::nn::Conv2dOptions(64, 64, 3).stride(2)),

              right_fc1_(GetConvOutput(input_shape), 16),
              right_fc2_(16, 8),
              right_fc3_(8, dof) {
            //   right_fc4_(8, dof) {

            // Left layers.
            register_module("left_conv1", left_conv1_);
            register_module("left_conv2", left_conv2_);
            register_module("left_conv3", left_conv3_);
            // register_module("left_conv4", left_conv4_);
            // register_module("left_conv5", left_conv5_);

            register_module("left_fc1", left_fc1_);
            register_module("left_fc2", left_fc2_);
            register_module("left_fc3", left_fc3_);
            // register_module("left_fc4", left_fc4_);

            // Right layers.
            register_module("right_conv1", right_conv1_);
            register_module("right_conv2", right_conv2_);
            register_module("right_conv3", right_conv3_);
            // register_module("right_conv4", right_conv4_);
            // register_module("right_conv5", right_conv5_);

            register_module("right_fc1", right_fc1_);
            register_module("right_fc2", right_fc2_);
            register_module("right_fc3", right_fc3_);
            // register_module("right_fc4", right_fc4_);
        }

        // Forward pass.
        torch::Tensor forward(torch::Tensor left_in, torch::Tensor right_in) {

            // Left layers.
            left_in = torch::relu(left_conv1_->forward(left_in));
            left_in = torch::relu(left_conv2_->forward(left_in));
            left_in = torch::relu(left_conv3_->forward(left_in));
            // left_in = torch::relu(left_conv4_->forward(left_in));
            // left_in = torch::relu(left_conv5_->forward(left_in));

            // Flatten.
            left_in = left_in.view({left_in.sizes()[0], -1});

            left_in = torch::relu(left_fc1_->forward(left_in));
            left_in = torch::relu(left_fc2_->forward(left_in));
            left_in = torch::relu(left_fc3_->forward(left_in));
            // left_in = torch::relu(left_fc4_->forward(left_in));

            // Right layers.
            right_in = torch::relu(right_conv1_->forward(right_in));
            right_in = torch::relu(right_conv2_->forward(right_in));
            right_in = torch::relu(right_conv3_->forward(right_in));
            // right_in = torch::relu(right_conv4_->forward(right_in));
            // right_in = torch::relu(right_conv5_->forward(right_in));

            // Flatten.
            right_in = right_in.view({right_in.sizes()[0], -1});

            right_in = torch::relu(right_fc1_->forward(right_in));
            right_in = torch::relu(right_fc2_->forward(right_in));
            right_in = torch::relu(right_fc3_->forward(right_in));
            // right_in = torch::relu(right_fc4_->forward(right_in));

            // Substract layers.
            return torch::tanh(left_in - right_in);
        }

    private:

        // Left layers.
        torch::nn::Conv2d left_conv1_;
        torch::nn::Conv2d left_conv2_;
        torch::nn::Conv2d left_conv3_;
        // torch::nn::Conv2d left_conv4_;
        // torch::nn::Conv2d left_conv5_;

        torch::nn::Linear left_fc1_;
        torch::nn::Linear left_fc2_;
        torch::nn::Linear left_fc3_;
        // torch::nn::Linear left_fc4_;

        // Right layers.
        torch::nn::Conv2d right_conv1_;
        torch::nn::Conv2d right_conv2_;
        torch::nn::Conv2d right_conv3_;
        // torch::nn::Conv2d right_conv4_;
        // torch::nn::Conv2d right_conv5_;

        torch::nn::Linear right_fc1_;
        torch::nn::Linear right_fc2_;
        torch::nn::Linear right_fc3_;
        // torch::nn::Linear right_fc4_;

        // Get number of elements of output.
        int64_t GetConvOutput(at::IntList input_shape) {

            torch::Tensor in = torch::zeros(input_shape, torch::kFloat).unsqueeze(0);
            torch::Tensor out = ForwardConv(in);

            return out.numel();
        }

        torch::Tensor ForwardConv(torch::Tensor in) {

            in = torch::relu(left_conv1_->forward(in));
            in = torch::relu(left_conv2_->forward(in));
            in = torch::relu(left_conv3_->forward(in));
            // in = torch::relu(left_conv4_->forward(in));
            // in = torch::relu(left_conv5_->forward(in));
            
            return in;
        };
};

// Implements a convolutional neural net for stereo inputs. It maps action-state pairs to Q-values.
class Critic : public torch::nn::Module {

    public:

            // Constructor.
        Critic(at::IntList input_shape /*{int64_t channels, int64_t height, int64_t width}*/, int64_t dof)
            : // Left layers.
              left_conv1_(torch::nn::Conv2dOptions(3, 8, 5).stride(2)),
              left_conv2_(torch::nn::Conv2dOptions(8, 16, 5).stride(2)),
              left_conv3_(torch::nn::Conv2dOptions(16, 32, 3).stride(2)),
            //   left_conv4_(torch::nn::Conv2dOptions(32, 64, 3).stride(2)),
            //   left_conv5_(torch::nn::Conv2dOptions(64, 64, 3).stride(2)),

              left_fc1_(GetConvOutput(input_shape), 16),
              left_fc2_(16, 8),
              left_fc3_(8, dof),
            //   left_fc4_(8, dof),

              // Right layers.
              right_conv1_(torch::nn::Conv2dOptions(3, 8, 5).stride(2)),
              right_conv2_(torch::nn::Conv2dOptions(8, 16, 5).stride(2)),
              right_conv3_(torch::nn::Conv2dOptions(16, 32, 3).stride(2)),
            //   right_conv4_(torch::nn::Conv2dOptions(32, 64, 3).stride(2)),
            //   right_conv5_(torch::nn::Conv2dOptions(64, 64, 3).stride(2)),

              right_fc1_(GetConvOutput(input_shape), 16),
              right_fc2_(16, 8),
              right_fc3_(8, dof),
            //   right_fc4_(8, dof),
              
              // Q-layer.
              fcq_(2*dof, 1) {

            // Left layers.
            register_module("left_conv1", left_conv1_);
            register_module("left_conv2", left_conv2_);
            register_module("left_conv3", left_conv3_);
            // register_module("left_conv4", left_conv4_);
            // register_module("left_conv5", left_conv5_);

            register_module("left_fc1", left_fc1_);
            register_module("left_fc2", left_fc2_);
            register_module("left_fc3", left_fc3_);
            // register_module("left_fc4", left_fc4_);

            // Right layers.
            register_module("right_conv1", right_conv1_);
            register_module("right_conv2", right_conv2_);
            register_module("right_conv3", right_conv3_);
            // register_module("right_conv4", right_conv4_);
            // register_module("right_conv5", right_conv5_);

            register_module("right_fc1", right_fc1_);
            register_module("right_fc2", right_fc2_);
            register_module("right_fc3", right_fc3_);
            // register_module("right_fc4", right_fc4_);

            // Q-layer.
            register_module("fcq", fcq_);
        }

        // Forward pass.
        torch::Tensor forward(torch::Tensor left_in, torch::Tensor right_in, torch::Tensor next_action) {

            // Left layers.
            left_in = torch::relu(left_conv1_->forward(left_in));
            left_in = torch::relu(left_conv2_->forward(left_in));
            left_in = torch::relu(left_conv3_->forward(left_in));
            // left_in = torch::relu(left_conv4_->forward(left_in));
            // left_in = torch::relu(left_conv5_->forward(left_in));

            // Flatten.
            left_in = left_in.view({left_in.sizes()[0], -1});            

            left_in = torch::relu(left_fc1_->forward(left_in));
            left_in = torch::relu(left_fc2_->forward(left_in));
            left_in = torch::relu(left_fc3_->forward(left_in));
            // left_in = torch::relu(left_fc4_->forward(left_in));

            // Right layers.
            right_in = torch::relu(right_conv1_->forward(right_in));
            right_in = torch::relu(right_conv2_->forward(right_in));
            right_in = torch::relu(right_conv3_->forward(right_in));
            // right_in = torch::relu(right_conv4_->forward(right_in));
            // right_in = torch::relu(right_conv5_->forward(right_in));

            // Flatten.
            right_in = right_in.view({right_in.sizes()[0], -1});

            right_in = torch::relu(right_fc1_->forward(right_in));
            right_in = torch::relu(right_fc2_->forward(right_in));
            right_in = torch::relu(right_fc3_->forward(right_in));
            // right_in = torch::relu(right_fc4_->forward(right_in));

            torch::Tensor out = torch::cat({torch::tanh(left_in - right_in), next_action}, 1);
            
            // Substract layers.
            return fcq_->forward(out);
        }

    private:

        // Left layers.
        torch::nn::Conv2d left_conv1_;
        torch::nn::Conv2d left_conv2_;
        torch::nn::Conv2d left_conv3_;
        // torch::nn::Conv2d left_conv4_;
        // torch::nn::Conv2d left_conv5_;

        torch::nn::Linear left_fc1_;
        torch::nn::Linear left_fc2_;
        torch::nn::Linear left_fc3_;
        // torch::nn::Linear left_fc4_;

        // Right layers.
        torch::nn::Conv2d right_conv1_;
        torch::nn::Conv2d right_conv2_;
        torch::nn::Conv2d right_conv3_;
        // torch::nn::Conv2d right_conv4_;
        // torch::nn::Conv2d right_conv5_;

        torch::nn::Linear right_fc1_;
        torch::nn::Linear right_fc2_;
        torch::nn::Linear right_fc3_;
        // torch::nn::Linear right_fc4_;
        
        // Q-layer.
        torch::nn::Linear fcq_;

        // Get number of elements of output.
        int64_t GetConvOutput(at::IntList input_shape) {

            torch::Tensor in = torch::zeros(input_shape, torch::kFloat).unsqueeze(0);
            torch::Tensor out = ForwardConv(in);

            return out.numel();
        }

        torch::Tensor ForwardConv(torch::Tensor in) {

            in = torch::relu(left_conv1_->forward(in));
            in = torch::relu(left_conv2_->forward(in));
            in = torch::relu(left_conv3_->forward(in));
            // in = torch::relu(left_conv4_->forward(in));
            // in = torch::relu(left_conv5_->forward(in));

            return in;
        };

};


// Implements a convolutional neural net for stereo inputs. It maps states to actions. Suitable for hybrid learning.
class DQNImpl : public torch::nn::Module {

    public:

        // Constructor.
        DQNImpl(int64_t channel, int64_t height, int64_t width, int64_t n_actions)
            : // Left layers.
              left_conv1_(register_module("left_conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 8, 5).stride(2)))),
              left_conv2_(register_module("left_conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(8, 16, 5).stride(2)))),
              left_conv3_(register_module("left_conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 3).stride(2)))),

              //left_bn1_(register_module("left_bn1", torch::nn::BatchNorm(8))),
              //left_bn2_(register_module("left_bn2", torch::nn::BatchNorm(16))),
              //left_bn3_(register_module("left_bn3", torch::nn::BatchNorm(32))),

              left_fc1_(register_module("left_fc1", torch::nn::Linear(GetConvOutput(channel, height, width), 16))),
              left_fc2_(register_module("left_fc2", torch::nn::Linear(16, 8))),
              left_fc3_(register_module("left_fc3", torch::nn::Linear(8, n_actions))),

              // Right layers.
              right_conv1_(register_module("right_conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 8, 5).stride(2)))),
              right_conv2_(register_module("right_conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(8, 16, 5).stride(2)))),
              right_conv3_(register_module("right_conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 3).stride(2)))),

              //right_bn1_(register_module("right_bn1", torch::nn::BatchNorm(8))),
              //right_bn2_(register_module("right_bn2", torch::nn::BatchNorm(16))),
              //right_bn3_(register_module("right_bn3", torch::nn::BatchNorm(32))),

              right_fc1_(register_module("right_fc1", torch::nn::Linear(GetConvOutput(channel, height, width), 16))),
              right_fc2_(register_module("right_fc2", torch::nn::Linear(16, 8))),
              right_fc3_(register_module("right_fc3", torch::nn::Linear(8, n_actions))) {

        };

        // Forward pass.
        torch::Tensor forward(torch::Tensor left_in, torch::Tensor right_in) {

            // Left layers.
            left_in = torch::relu(left_conv1_(left_in));
            left_in = torch::relu(left_conv2_(left_in));
            left_in = torch::relu(left_conv3_(left_in));

            // Flatten.
            left_in = left_in.view({left_in.sizes()[0], -1});

            left_in = torch::relu(left_fc1_(left_in));
            left_in = torch::relu(left_fc2_(left_in));
            left_in = torch::relu(left_fc3_(left_in));

            // Right layers.
            right_in = torch::relu(right_conv1_(right_in));
            right_in = torch::relu(right_conv2_(right_in));
            right_in = torch::relu(right_conv3_(right_in));

            // Flatten.
            right_in = right_in.view({right_in.sizes()[0], -1});

            right_in = torch::relu(right_fc1_(right_in));
            right_in = torch::relu(right_fc2_(right_in));
            right_in = torch::relu(right_fc3_(right_in));

            // Substract layers.
            return left_in - right_in;
        };

    private:

        // Left layers.
        torch::nn::Conv2d left_conv1_;
        torch::nn::Conv2d left_conv2_;
        torch::nn::Conv2d left_conv3_;

        //torch::nn::BatchNorm left_bn1_;
        //torch::nn::BatchNorm left_bn2_;
        //torch::nn::BatchNorm left_bn3_;

        torch::nn::Linear left_fc1_;
        torch::nn::Linear left_fc2_;
        torch::nn::Linear left_fc3_;

        // Right layers.
        torch::nn::Conv2d right_conv1_;
        torch::nn::Conv2d right_conv2_;
        torch::nn::Conv2d right_conv3_;

        //torch::nn::BatchNorm right_bn1_;
        //torch::nn::BatchNorm right_bn2_;
        //torch::nn::BatchNorm right_bn3_;

        torch::nn::Linear right_fc1_;
        torch::nn::Linear right_fc2_;
        torch::nn::Linear right_fc3_;

        // Get number of elements of output.
        int64_t GetConvOutput(int64_t channel, int64_t height, int64_t width) {

            torch::Tensor in = torch::zeros({channel, height, width}, torch::kFloat).unsqueeze(0);
            torch::Tensor out = ForwardConv(in);

            return out.numel();
        };

        torch::Tensor ForwardConv(torch::Tensor in) {

            in = torch::relu(left_conv1_->forward(in));
            in = torch::relu(left_conv2_->forward(in));
            in = torch::relu(left_conv3_->forward(in));
            
            return in;
        };
};

TORCH_MODULE(DQN);

#endif // MODELS_H_
