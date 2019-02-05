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
              left_conv4_(torch::nn::Conv2dOptions(32, 64, 3).stride(2)),
              left_conv5_(torch::nn::Conv2dOptions(64, 64, 3).stride(2)),

              left_fc1_(GetConvOutput(input_shape), 32),
              left_fc2_(32, 16),
              left_fc3_(16, 8),
              left_fc4_(8, dof),

              // Right layers.
              right_conv1_(torch::nn::Conv2dOptions(3, 8, 5).stride(2)),
              right_conv2_(torch::nn::Conv2dOptions(8, 16, 5).stride(2)),
              right_conv3_(torch::nn::Conv2dOptions(16, 32, 3).stride(2)),
              right_conv4_(torch::nn::Conv2dOptions(32, 64, 3).stride(2)),
              right_conv5_(torch::nn::Conv2dOptions(64, 64, 3).stride(2)),

              right_fc1_(GetConvOutput(input_shape), 32),
              right_fc2_(32, 16),
              right_fc3_(16, 8),
              right_fc4_(8, dof) {

            // Left layers.
            register_module("left_conv1", left_conv1_);
            register_module("left_conv2", left_conv2_);
            register_module("left_conv3", left_conv3_);
            register_module("left_conv4", left_conv4_);
            register_module("left_conv5", left_conv5_);

            register_module("left_fc1", left_fc1_);
            register_module("left_fc2", left_fc2_);
            register_module("left_fc3", left_fc3_);
            register_module("left_fc4", left_fc4_);

            // Right layers.
            register_module("right_conv1", right_conv1_);
            register_module("right_conv2", right_conv2_);
            register_module("right_conv3", right_conv3_);
            register_module("right_conv4", right_conv4_);
            register_module("right_conv5", right_conv5_);

            register_module("right_fc1", right_fc1_);
            register_module("right_fc2", right_fc2_);
            register_module("right_fc3", right_fc3_);
            register_module("right_fc4", right_fc4_);
        }

        // Forward pass.
        torch::Tensor forward(torch::Tensor left_in, torch::Tensor right_in) {

            // Left layers.
            left_in = torch::relu(left_conv1_->forward(left_in));
            left_in = torch::relu(left_conv2_->forward(left_in));
            left_in = torch::relu(left_conv3_->forward(left_in));
            left_in = torch::relu(left_conv4_->forward(left_in));
            left_in = torch::relu(left_conv5_->forward(left_in));

            // Flatten.
            left_in = left_in.view({left_in.sizes()[0], -1});

            left_in = torch::relu(left_fc1_->forward(left_in));
            std::cout << "size" << left_in.sizes() << std::endl;
            left_in = torch::relu(left_fc2_->forward(left_in));
            left_in = torch::relu(left_fc3_->forward(left_in));
            left_in = torch::relu(left_fc4_->forward(left_in));

            // Right layers.
            right_in = torch::relu(right_conv1_->forward(right_in));
            right_in = torch::relu(right_conv2_->forward(right_in));
            right_in = torch::relu(right_conv3_->forward(right_in));
            right_in = torch::relu(right_conv4_->forward(right_in));
            right_in = torch::relu(right_conv5_->forward(right_in));

            // Flatten.
            right_in = right_in.view({left_in.sizes()[0], -1});

            right_in = torch::relu(right_fc1_->forward(right_in));
            right_in = torch::relu(right_fc2_->forward(right_in));
            right_in = torch::relu(right_fc3_->forward(right_in));
            right_in = torch::relu(right_fc4_->forward(right_in));

            // Substract layers.
            return left_in - right_in;
        }

    private:

        // Left layers.
        torch::nn::Conv2d left_conv1_;
        torch::nn::Conv2d left_conv2_;
        torch::nn::Conv2d left_conv3_;
        torch::nn::Conv2d left_conv4_;
        torch::nn::Conv2d left_conv5_;

        torch::nn::Linear left_fc1_;
        torch::nn::Linear left_fc2_;
        torch::nn::Linear left_fc3_;
        torch::nn::Linear left_fc4_;

        // Right layers.
        torch::nn::Conv2d right_conv1_;
        torch::nn::Conv2d right_conv2_;
        torch::nn::Conv2d right_conv3_;
        torch::nn::Conv2d right_conv4_;
        torch::nn::Conv2d right_conv5_;

        torch::nn::Linear right_fc1_;
        torch::nn::Linear right_fc2_;
        torch::nn::Linear right_fc3_;
        torch::nn::Linear right_fc4_;

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
            in = torch::relu(left_conv4_->forward(in));
            in = torch::relu(left_conv5_->forward(in));
            
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
              left_conv4_(torch::nn::Conv2dOptions(32, 64, 3).stride(2)),
              left_conv5_(torch::nn::Conv2dOptions(64, 64, 3).stride(2)),

              left_fc1_(GetConvOutput(input_shape), 32),
              left_fc2_(32, 16),
              left_fc3_(16, 8),
              left_fc4_(8, dof),

              // Right layers.
              right_conv1_(torch::nn::Conv2dOptions(3, 8, 5).stride(2)),
              right_conv2_(torch::nn::Conv2dOptions(8, 16, 5).stride(2)),
              right_conv3_(torch::nn::Conv2dOptions(16, 32, 3).stride(2)),
              right_conv4_(torch::nn::Conv2dOptions(32, 64, 3).stride(2)),
              right_conv5_(torch::nn::Conv2dOptions(64, 64, 3).stride(2)),

              right_fc1_(GetConvOutput(input_shape), 32),
              right_fc2_(32, 16),
              right_fc3_(16, 8),
              right_fc4_(8, dof),
              
              // Q-layer.
              fcq_(dof, 1) {

            // Left layers.
            register_module("left_conv1", left_conv1_);
            register_module("left_conv2", left_conv2_);
            register_module("left_conv3", left_conv3_);
            register_module("left_conv4", left_conv4_);
            register_module("left_conv5", left_conv5_);

            register_module("left_fc1", left_fc1_);
            register_module("left_fc2", left_fc2_);
            register_module("left_fc3", left_fc3_);
            register_module("left_fc4", left_fc4_);

            // Right layers.
            register_module("right_conv1", right_conv1_);
            register_module("right_conv2", right_conv2_);
            register_module("right_conv3", right_conv3_);
            register_module("right_conv4", right_conv4_);
            register_module("right_conv5", right_conv5_);

            register_module("right_fc1", right_fc1_);
            register_module("right_fc2", right_fc2_);
            register_module("right_fc3", right_fc3_);
            register_module("right_fc4", right_fc4_);

            // Q-layer.
            register_module("fcq", fcq_);
        }

        // Forward pass.
        torch::Tensor forward(torch::Tensor left_in, torch::Tensor right_in, torch::Tensor state) {

            // Left layers.
            left_in = torch::relu(left_conv1_->forward(left_in));
            left_in = torch::relu(left_conv2_->forward(left_in));
            left_in = torch::relu(left_conv3_->forward(left_in));
            left_in = torch::relu(left_conv4_->forward(left_in));
            left_in = torch::relu(left_conv5_->forward(left_in));

            // Flatten.
            left_in = left_in.view({left_in.sizes()[0], -1});            

            left_in = torch::relu(left_fc1_->forward(left_in));
            left_in = torch::relu(left_fc2_->forward(left_in));
            left_in = torch::relu(left_fc3_->forward(left_in));
            left_in = torch::relu(left_fc4_->forward(left_in));

            // Right layers.
            right_in = torch::relu(right_conv1_->forward(right_in));
            right_in = torch::relu(right_conv2_->forward(right_in));
            right_in = torch::relu(right_conv3_->forward(right_in));
            right_in = torch::relu(right_conv4_->forward(right_in));
            right_in = torch::relu(right_conv5_->forward(right_in));

            right_in = torch::relu(right_fc1_->forward(right_in));
            right_in = torch::relu(right_fc2_->forward(right_in));
            right_in = torch::relu(right_fc3_->forward(right_in));
            right_in = torch::relu(right_fc4_->forward(right_in));

            // Flatten.
            right_in = right_in.view({left_in.sizes()[0], -1});

            // Concatenate state and actions.
            torch::Tensor out = torch::cat({left_in - right_in, state}, 0);
            

            // Substract layers.
            return fcq_->forward(out);
        }

    private:

        // Left layers.
        torch::nn::Conv2d left_conv1_;
        torch::nn::Conv2d left_conv2_;
        torch::nn::Conv2d left_conv3_;
        torch::nn::Conv2d left_conv4_;
        torch::nn::Conv2d left_conv5_;

        torch::nn::Linear left_fc1_;
        torch::nn::Linear left_fc2_;
        torch::nn::Linear left_fc3_;
        torch::nn::Linear left_fc4_;

        // Right layers.
        torch::nn::Conv2d right_conv1_;
        torch::nn::Conv2d right_conv2_;
        torch::nn::Conv2d right_conv3_;
        torch::nn::Conv2d right_conv4_;
        torch::nn::Conv2d right_conv5_;

        torch::nn::Linear right_fc1_;
        torch::nn::Linear right_fc2_;
        torch::nn::Linear right_fc3_;
        torch::nn::Linear right_fc4_;
        
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
            in = torch::relu(left_conv4_->forward(in));
            in = torch::relu(left_conv5_->forward(in));

            return in;
        };

};

#endif // MODELS_H_
