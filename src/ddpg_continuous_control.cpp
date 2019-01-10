#include "ddpg_continuous_control.h"

#define LR_ACTOR 1e-4    // Learning rate.
#define LR_CRITIC 1e-3   // Learning rate.
#define WEIGHT_DECAY 0.  // Weigt decay.
#define GAMMA 0.99       // Discount factor.

DDPGContinuousControl::DDPGContinuousControl(at::IntList input_shape, int64_t dof, int64_t batch_size, int64_t buffer_size)
    : batch_size_(batch_size),
      
      actor_local_(input_shape, dof, batch_size),
      actor_target_(input_shape, dof, batch_size),
      actor_opt_(actor_local_.parameters(), torch::optim::AdamOptions(LR_ACTOR).weight_decay(WEIGHT_DECAY)),
      
      critic_local_(input_shape, dof, batch_size),
      critic_target_(input_shape, dof, batch_size),
      critic_opt_(critic_local_.parameters(), torch::optim::AdamOptions(LR_CRITIC).weight_decay(WEIGHT_DECAY)),
      
      replay_memory_(buffer_size, batch_size) {

};

void DDPGContinuousControl::Step(memory& states) {

    // Save experience in replay memory, and use random sample from buffer to learn.
    for (int i = 0; i < states.size(); i++) {

        replay_memory_.Add(states[i]);
    }

    // Learn if enough samples are available.
    if (replay_memory_.Length() > batch_size_) {

        memoryptr replay_memory_ptr = replay_memory_.Sample();
        
        Learn(replay_memory_ptr, GAMMA);
    }
}

void DDPGContinuousControl::Act(torch::Tensor left_in, torch::Tensor right_in, bool add_noise) {

    // Returns actions for a given input as per current policy.
    actor_local_.eval();

    torch::NoGradGuard no_grad;

    torch::Tensor action = actor_local_.forward(left_in, right_in);

    actor_local_.train();

    if (add_noise) {

        // action += add some noise
    }
}

void DDPGContinuousControl::Learn(memoryptr states, double gamma) {

    // 
}
