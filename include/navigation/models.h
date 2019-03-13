#ifndef MODELS_H_
#define MODELS_H_ 

#include <torch/torch.h>
#include <math.h>

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

// Network model for Proximal Policy Optimization with Nonlinear Model Predictive Control.
struct ActorCriticImpl : public torch::nn::Module 
{
    // Actor.
    torch::Tensor mu_;
    torch::Tensor std_;

    // Left layers.
    torch::nn::Conv2d a_left_conv1_, a_left_conv2_, a_left_conv3_;
    torch::nn::Linear a_left_fc1_, a_left_fc2_, a_left_fc3_;

    // Right layers.
    torch::nn::Conv2d a_right_conv1_, a_right_conv2_, a_right_conv3_;
    torch::nn::Linear a_right_fc1_, a_right_fc2_, a_right_fc3_;

    // Critic.
    torch::nn::Linear c_val_;

    // Left layers.
    torch::nn::Conv2d c_left_conv1_, c_left_conv2_, c_left_conv3_;
    torch::nn::Linear c_left_fc1_, c_left_fc2_, c_left_fc3_;

    // Right layers.
    torch::nn::Conv2d c_right_conv1_, c_right_conv2_, c_right_conv3_;
    torch::nn::Linear c_right_fc1_, c_right_fc2_, c_right_fc3_;

    ActorCriticImpl(int64_t channel, int64_t height, int64_t width, int64_t n_actions, double std)
        : // Actor.
          mu_(torch::full(n_actions, 0.)),
          std_(torch::full(n_actions, std, torch::kFloat64)),

          // Left layers.
          a_left_conv1_(torch::nn::Conv2dOptions(3, 8, 5).stride(2)), 
          a_left_conv2_(torch::nn::Conv2dOptions(8, 16, 5).stride(2)), 
          a_left_conv3_(torch::nn::Conv2dOptions(16, 32, 3).stride(2)),
          
          a_left_fc1_(GetConvOutput(channel, height, width), 16), 
          a_left_fc2_(16, 8), 
          a_left_fc3_(8, n_actions),

          // Right layers.
          a_right_conv1_(torch::nn::Conv2dOptions(3, 8, 5).stride(2)), 
          a_right_conv2_(torch::nn::Conv2dOptions(8, 16, 5).stride(2)), 
          a_right_conv3_(torch::nn::Conv2dOptions(16, 32, 3).stride(2)),

          a_right_fc1_(GetConvOutput(channel, height, width), 16), 
          a_right_fc2_(16, 8), 
          a_right_fc3_(8, n_actions),
          
          // Critic
          c_val_(torch::nn::Linear(n_actions, 1)),

          // Left layers.
          c_left_conv1_(torch::nn::Conv2dOptions(3, 8, 5).stride(2)), 
          c_left_conv2_(torch::nn::Conv2dOptions(8, 16, 5).stride(2)), 
          c_left_conv3_(torch::nn::Conv2dOptions(16, 32, 3).stride(2)),
          
          c_left_fc1_(GetConvOutput(channel, height, width), 16), 
          c_left_fc2_(16, 8), 
          c_left_fc3_(8, n_actions),

          // Right lcyers.
          c_right_conv1_(torch::nn::Conv2dOptions(3, 8, 5).stride(2)), 
          c_right_conv2_(torch::nn::Conv2dOptions(8, 16, 5).stride(2)), 
          c_right_conv3_(torch::nn::Conv2dOptions(16, 32, 3).stride(2)),

          c_right_fc1_(GetConvOutput(channel, height, width), 16), 
          c_right_fc2_(16, 8), 
          c_right_fc3_(8, n_actions)
    {
        // Register the modules and parameters.
        // Actor.
        register_parameter("std", std_);

        // Left layers.
        register_module("a_left_conv1", a_left_conv1_);
        register_module("a_left_conv2", a_left_conv2_);
        register_module("a_left_conv3", a_left_conv3_);

        register_module("a_left_fc1", a_left_fc1_);
        register_module("a_left_fc2", a_left_fc2_);
        register_module("a_left_fc3", a_left_fc3_);

        // Right layers.
        register_module("a_right_conv1", a_right_conv1_);
        register_module("a_right_conv2", a_right_conv2_);
        register_module("a_right_conv3", a_right_conv3_);

        register_module("a_right_fc1", a_right_fc1_);
        register_module("a_right_fc2", a_right_fc2_);
        register_module("a_right_fc3", a_right_fc3_);

        // Critic.
        register_module("c_val", c_val_);

        // Left layers.
        register_module("a_left_conv1", a_left_conv1_);
        register_module("a_left_conv2", a_left_conv2_);
        register_module("a_left_conv3", a_left_conv3_);

        register_module("a_left_fc1", a_left_fc1_);
        register_module("a_left_fc2", a_left_fc2_);
        register_module("a_left_fc3", a_left_fc3_);

        // Right layers.
        register_module("a_right_conv1", a_right_conv1_);
        register_module("a_right_conv2", a_right_conv2_);
        register_module("a_right_conv3", a_right_conv3_);

        register_module("a_right_fc1", a_right_fc1_);
        register_module("a_right_fc2", a_right_fc2_);
        register_module("a_right_fc3", a_right_fc3_);
    }

    // Forward pass.
    auto forward(torch::Tensor left_in, torch::Tensor right_in) -> std::tuple<torch::Tensor, torch::Tensor> 
    {

        // Actor.
        // Left layers.
        torch::Tensor a_left = torch::relu(a_left_conv1_->forward(left_in));
        a_left = torch::relu(a_left_conv2_->forward(a_left));
        a_left = torch::tanh(a_left_conv3_->forward(a_left));

        // Flatten.
        a_left = a_left.view({a_left.sizes()[0], -1});

        a_left = torch::relu(a_left_fc1_->forward(a_left));
        a_left = torch::relu(a_left_fc2_->forward(a_left));
        a_left = torch::relu(a_left_fc3_->forward(a_left));

        // Right layers.
        torch::Tensor a_right = torch::relu(a_right_conv1_->forward(right_in));
        a_right = torch::relu(a_right_conv2_->forward(a_right));
        a_right = torch::tanh(a_right_conv3_->forward(a_right));

        // Flatten.
        a_right = right_in.view({a_right.sizes()[0], -1});

        a_right = torch::relu(a_right_fc1_->forward(a_right));
        a_right = torch::relu(a_right_fc2_->forward(a_right));
        a_right = torch::relu(a_right_fc3_->forward(a_right));

        mu_ = (a_left + a_right).div(2.);

        // Critic.
        // Left layers.
        torch::Tensor c_left = torch::relu(c_left_conv1_->forward(left_in));
        c_left = torch::relu(c_left_conv2_->forward(c_left));
        c_left = torch::tanh(c_left_conv3_->forward(c_left));

        // Flatten.
        c_left = c_left.view({c_left.sizes()[0], -1});

        c_left = torch::relu(c_left_fc1_->forward(c_left));
        c_left = torch::relu(c_left_fc2_->forward(c_left));
        c_left = torch::relu(c_left_fc3_->forward(c_left));

        // Right layers.
        torch::Tensor c_right = torch::relu(c_right_conv1_->forward(right_in));
        c_right = torch::relu(c_right_conv2_->forward(c_right));
        c_right = torch::tanh(c_right_conv3_->forward(c_right));

        // Flatten.
        c_right = c_right.view({c_right.sizes()[0], -1});

        c_right = torch::relu(c_right_fc1_->forward(c_right));
        c_right = torch::relu(c_right_fc2_->forward(c_right));
        c_right = torch::relu(c_right_fc3_->forward(c_right));

        torch::Tensor val = (c_left + c_right).div(2.);

        // Value layer.
        val = c_val_->forward(val);

        if (this->is_training()) 
        {
            torch::NoGradGuard no_grad;

            torch::Tensor action = torch::normal(mu_, std_.abs().expand_as(mu_));
            return std::make_tuple(action, val);  
        }
        else 
        {
            return std::make_tuple(mu_, val);  
        }
    }

    // Initialize network.
    void normal(double mu, double std) 
    {
        torch::NoGradGuard no_grad;

        for (auto& p: this->parameters()) 
        {
            p.normal_(mu,std);
        }         
    }

    void zero() 
    {
        torch::NoGradGuard no_grad;

        for (auto& p: this->parameters()) 
        {
            p.zero_();
        }         
    }

    auto entropy() -> torch::Tensor
    {
        // Differential entropy of normal distribution. For reference https://pytorch.org/docs/stable/_modules/torch/distributions/normal.html#Normal
        return 0.5 + 0.5*log(2*M_PI) + std_.abs().log();
    }

    auto log_prob(torch::Tensor action) -> torch::Tensor
    {
        // Logarithmic probability of taken action, given the current distribution.
        torch::Tensor var = std_*std_;
        torch::Tensor log_scale = std_.abs().log();

        return -((action - mu_)*(action - mu_))/(2*var) - log_scale - log(sqrt(2*M_PI));
    }

    // Get number of elements of output.
    int64_t GetConvOutput(int64_t channel, int64_t height, int64_t width) {

        torch::Tensor in = torch::zeros({channel, height, width}, torch::kFloat).unsqueeze(0);
        torch::Tensor out = ForwardConv(in);

        return out.numel();
    };

    torch::Tensor ForwardConv(torch::Tensor in) {

        in = torch::relu(a_left_conv1_->forward(in));
        in = torch::relu(a_left_conv2_->forward(in));
        in = torch::relu(a_left_conv3_->forward(in));
        
        return in;
    };
};

TORCH_MODULE(ActorCritic);

#endif // MODELS_H_
