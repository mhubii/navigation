#include "q_learning.h"

#define GAMMA 0.99f
#define TARGET_UPDATE 100
#define EPS_START 0.9f
#define EPS_END 0.05f
#define EPS_DECAY 1000

QLearning::QLearning(int64_t channel, int64_t height, int64_t width, int64_t n_actions, int64_t batch_size, int64_t buffer_size, torch::Device device)
        : rd_(),
          random_engine_(rd_()),
        
          n_actions_(n_actions),
          batch_size_(batch_size),
          device_(device),
          
          policy_(channel, height, width, n_actions),
          target_(channel, height, width, n_actions),
          opt_(policy_->parameters(), torch::optim::AdamOptions(0.0001)),

          n_update_(0),
          n_steps_(0),
          
          replay_memory_(buffer_size, batch_size),
          states_batch_({}),
          
          loss_(torch::zeros({}, device)) {

        policy_->to(device_);
        target_->to(device_);
}

void QLearning::Step(state& state) {

        // Save experience in replay memory, and use random sample from buffer to learn.
        replay_memory_.Add(state);

        // Learn if enough samples are available.
        if (replay_memory_.Length() > batch_size_) {

                states_batch_ = replay_memory_.Sample(device_); // TODO changes order or something

                if (states_batch_) {
                        
                        Learn(states_batch_, GAMMA);
                        n_update_ += 1;
                }

                else {

                        printf("QLearning -- no state batches received.\n");
                }
        }    

        if (n_update_ % TARGET_UPDATE == 0) {

                // Copy policy to target net.
                torch::NoGradGuard no_grad;
                
                for (uint i = 0; i < target_->parameters().size(); i++) {

                        target_->parameters()[i].copy_(policy_->parameters()[i]);
                }
        }   
}

torch::Tensor QLearning::Act(torch::Tensor left_in, torch::Tensor right_in, bool train) {

        // New step and update randomness of action.
        n_steps_ += 1;
        float eps = EPS_END + (EPS_START - EPS_END)*std::exp(-(float)n_steps_/EPS_DECAY);

        // Get action with highest future expected reward policy with some randomness.
        float val = std::uniform_real_distribution<float>(0., 1.)(random_engine_);

        if (val > eps) {

                torch::NoGradGuard no_grad;

                return std::get<1>(policy_->forward(left_in.to(device_), right_in.to(device_)).max(1)).view({1,1});
        }

        else {
        
                return torch::full({1,1}, std::uniform_int_distribution<int>(0,n_actions_-1)(random_engine_), torch::kLong);
        }
}

void QLearning::Learn(states_batch& states, float gamma) {

        // Extract states, actions, and rewards.
        torch::Tensor& l_imgs = states.left_imgs;
        torch::Tensor& r_imgs = states.right_imgs;
        torch::Tensor& actions = states.actions;
        torch::Tensor& rewards = states.rewards;
        torch::Tensor& l_imgs_next = states.next_left_imgs;
        torch::Tensor& r_imgs_next = states.next_right_imgs;
        torch::Tensor& dones = states.dones;

        // Use policy net to pick action with highest reward.
        torch::Tensor q_policy = policy_->forward(l_imgs, r_imgs).gather(1, actions); // expected rewards for actions

        // Predict reward of taken actions.
        torch::Tensor q_target = rewards + gamma*(std::get<0>(target_->forward(l_imgs_next, r_imgs_next).max(1)).unsqueeze(1).detach());//*(1-dones));

        // Huber loss and optimize.
        loss_ = torch::smooth_l1_loss(q_policy, q_target);
        opt_.zero_grad();
        loss_.backward();

        // Clamp parameters.
        for (uint i = 0; i < policy_->parameters().size(); i++) {

                torch::clamp(policy_->parameters()[i], -1., 1.);
        }

        opt_.step();
}
