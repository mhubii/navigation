#ifndef DDPG_CONTINUOUS_CONTROL_H_
#define DDPG_CONTINUOUS_CONTROL_H_

#include <torch/torch.h>

#include "models.h"
#include "ornstein_uhlenbeck_process.h"
#include "replay_memory.h"

class DDPGContinuousControl {

    public:

        DDPGContinuousControl(torch::IntArrayRef input_shape, int64_t dof, int64_t batch_size, int64_t buffer_size, torch::Device device);

        void Step(state& state);

        torch::Tensor Act(torch::Tensor left_in, torch::Tensor right_in, bool add_noise);

        void Reset();

        void Learn(states_batch& states, double gamma);

        void SoftUpdate(torch::nn::Module& local_model, torch::nn::Module& target_model, double tau);

    private:

        int64_t batch_size_;

        // Device type, CPU/GPU.
        torch::Device device_;

        // Ornstein-Uhlenbeck process.
        OUProcess ou_process_;

        // Networks and related stuff.
        Actor actor_local_;
        Actor actor_target_;
        torch::optim::Adam actor_opt_;

        Critic critic_local_;
        Critic critic_target_;
        torch::optim::Adam critic_opt_;

        ReplayMemory replay_memory_;
        states_batch states_batch_;
};

#endif