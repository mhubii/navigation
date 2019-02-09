#include "q_learning.h"

#define GAMMA 0.99f
#define TARGET_UPDATE 10
#define EPS_START 0.9f
#define EPS_END 0.05f
#define EPS_DECAY 200

QLearning::QLearning(torch::IntList input_shape, int64_t n_actions, int64_t batch_size, int64_t buffer_size, torch::Device device)
        : rd_(),
          random_engine_(rd_),
        
          n_actions_(n_actions),
          batch_size_(batch_size),
          device_(device),
          
          policy_(input_shape, n_actions),
          target_(input_shape, n_actions),
          opt_(policy_.parameters(), torch::optim::AdamOptions(0.001)),

          n_update_(0),
          n_steps_(0),
          
          replay_memory_(buffer_size, batch_size),
          states_batch_({}) {

}

void QLearning::Step(state& state) {

        // Save experience in replay memory, and use random sample from buffer to learn.
        replay_memory_.Add(state);

        // Learn if enough samples are available.
        if (replay_memory_.Length() > batch_size_) {

                states_batch_ = replay_memory_.Sample(device_);

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
        }   
}

torch::Tensor QLearning::Act(torch::Tensor left_in, torch::Tensor right_in, bool train) {

        // New step and update randomness of action.
        n_steps_ += 1;
        float eps = EPS_END + (EPS_START - EPS_END)*exp(-n_steps_/EPS_DECAY);

        // Get action with highest future expected reward policy with some randomness.
        float val = std::uniform_real_distribution<float>(0., 1.)(random_engine_);

        if (val > eps) {

                torch::NoGradGuard();

                return std::get<1>(policy_.forward(left_in, right_in).max(1));
        }

        else {

                return torch::full({1,1}, std::uniform_int_distribution<int>(0,n_actions_)(random_engine));
        }
}

void QLearning::Learn(states_batch& states, double gamma) {

        // Extract states, actions, and rewards.
        torch::Tensor& l_imgs = states.left_imgs;
        torch::Tensor& r_imgs = states.right_imgs;
        torch::Tensor& actions = states.actions;
        torch::Tensor& rewards = states.rewards;
        torch::Tensor& l_imgs_next = states.next_left_imgs;
        torch::Tensor& r_imgs_next = states.next_right_imgs;
        torch::Tensor& dones = states.dones;

        // Use policy net to pick action with highest reward.
        torch::Tensor q_policy = policy_.forward(l_imgs, r_imgs);

        // Predict reward of taken actions.
        //torch::Tensor q_next = rewards + gamma*policy_(l_imgs_next, r_imgs_next)

        //torch::Tensor q_
}
