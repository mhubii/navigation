#include "ddpg_continuous_control.h"

#define LR_ACTOR 1e-4    // Learning rate.
#define LR_CRITIC 1e-3   // Learning rate.
#define WEIGHT_DECAY 0.  // Weigt decay.
#define GAMMA 0.99       // Discount factor.
#define TAU 1e-3         // For soft update of target parameters.

#define MU 0.      // Mean of the Ornstein-Uhlenbeck process.
#define THETA 0.15 // Uncertainty parameters of the Ornstein-Uhlenbeck process.
#define SIGMA 0.2

DDPGContinuousControl::DDPGContinuousControl(at::IntList input_shape, int64_t dof, int64_t batch_size, int64_t buffer_size)
    : batch_size_(batch_size),

      ou_process_({batch_size, 1, dof}, MU, THETA, SIGMA), // size of ou_process?
      
      actor_local_(input_shape, dof),
      actor_target_(input_shape, dof),
      actor_opt_(actor_local_.parameters(), torch::optim::AdamOptions(LR_ACTOR).weight_decay(WEIGHT_DECAY)),
      
      critic_local_(input_shape, dof),
      critic_target_(input_shape, dof),
      critic_opt_(critic_local_.parameters(), torch::optim::AdamOptions(LR_CRITIC).weight_decay(WEIGHT_DECAY)),
      
      replay_memory_(buffer_size, batch_size),
      states_batch_({}) {

};

void DDPGContinuousControl::Step(memory& states) {

    // // Save experience in replay memory, and use random sample from buffer to learn.
    for (int i = 0; i < states.size(); i++) {

        replay_memory_.Add(states[i]);
    }

    // Learn if enough samples are available.
    if (replay_memory_.Length() > batch_size_) {

        states_batch_ = replay_memory_.Sample();
        
        if (states_batch_) {
        
            Learn(states_batch_, GAMMA);
        }

        else {

            printf("DDPGContinuousControl -- no state batches received.\n");
        }
    }
}

void DDPGContinuousControl::Act(torch::Tensor left_in, torch::Tensor right_in, bool add_noise) {

    // Returns actions for a given input as per current policy.
    actor_local_.eval();

    torch::NoGradGuard no_grad;

    torch::Tensor action = actor_local_.forward(left_in, right_in);

    actor_local_.train();

    if (add_noise) {

        action += ou_process_.Sample();
    }
}

void DDPGContinuousControl::Reset() {

    // TODO
}

void DDPGContinuousControl::Learn(states_batch& states, double gamma) {

    // Update policy and value paramters using given batch of memory tuples.
    // Q_targets = r + y * critic_target(next_state, actor_target(next_state))
    // where:
    //     actor_target(state) -> action
    //     critic_target(state, action) -> Q-value
    torch::Tensor& left_imgs = states.left_imgs;
    torch::Tensor& right_imgs = states.right_imgs;
    torch::Tensor& actions = states.actions;
    torch::Tensor& rewards = states.rewards;
    torch::Tensor& next_left_imgs = states.next_left_imgs;
    torch::Tensor& next_right_imgs = states.next_left_imgs;
    torch::Tensor& dones = states.dones;

    // Update critic.
    // Get predicted next-state actoins and Q-values from target models.
    torch::Tensor actions_next = actor_target_.forward(next_left_imgs, next_right_imgs);
    torch::Tensor q_targets_next = critic_target_.forward(next_left_imgs, next_right_imgs, actions_next);

    // Compute Q-targets for current states.
    torch::Tensor q_targets = rewards + gamma*q_targets_next*(1 - dones);

    // Compute critic loss.
    torch::Tensor q_expected = critic_local_.forward(left_imgs, right_imgs, actions);
    torch::Tensor critic_loss = torch::mse_loss(q_expected, q_targets);

    // Minimize loss.
    critic_opt_.zero_grad();
    critic_loss.backward();

    // Clip gradient norm not implemented atm.. It is kind of! https://pytorch.org/cppdocs/api/function_namespaceat_1a20be2de650096933b7c01c4d943fda11.html?highlight=clamp
    for (uint i = 0; i < critic_local_.parameters().size(); i++) {
    
        torch::clamp(critic_local_.parameters()[i], 0., 1.); 
    }

    critic_opt_.step();

    // Update actor.
    // Compute actor loss.
    torch::Tensor actions_pred = actor_local_.forward(left_imgs, right_imgs);
    torch::Tensor actor_loss = -critic_local_.forward(left_imgs, right_imgs, actions_pred).mean();
    
    // Minimize the loss.
    actor_opt_.zero_grad();
    actor_loss.backward();
    actor_opt_.step();

    // Update target networks.
    SoftUpdate(critic_local_, critic_target_, TAU);
    SoftUpdate(actor_local_, actor_target_, TAU);
}

void DDPGContinuousControl::SoftUpdate(torch::nn::Module& local_model, torch::nn::Module& target_model, double tau) {

    // Soft update model parameters. Iterate over vector of tensors.
    for (uint i = 0; i < local_model.parameters().size(); i++) {

        target_model.parameters().at(i).set_data(tau*local_model.parameters().at(i) + (1. - tau)*target_model.parameters().at(i));
    }
}
