#ifndef Q_LEARNING_H_
#define Q_LEARNING_H_

#include <torch/torch.h>

#include "models.h"
#include "replay_memory.h"


class QLearning {

    public:

        QLearning(int64_t channel, int64_t height, int64_t width, int64_t n_actions, int64_t batch_size, int64_t buffer_size, torch::Device device);

        void Step(state& state);

        torch::Tensor Act(torch::Tensor left_in, torch::Tensor right_in, bool train);

        void Learn(states_batch& states, float gamma);

        inline const DQN& GetTarget() const { return target_; };

        void SetPolicy(DQN& dqn);

        inline const float GetLoss() const { return *(loss_.to(torch::kCPU).data<float>()); };

    private:

        // Random device and random engine.
        std::random_device rd_;
        std::mt19937 random_engine_;

        int64_t n_actions_;
        int64_t batch_size_;

        // Device type, CPU/GPU.
        torch::Device device_;

        // Network and related stuff.
        DQN policy_;
        DQN target_;

        torch::optim::Adam opt_;

        uint n_update_;
        uint n_steps_;

        ReplayMemory replay_memory_;
        states_batch states_batch_;

        // Loss for one batch.
        torch::Tensor loss_;
};

#endif
