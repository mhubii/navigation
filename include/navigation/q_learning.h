#pragma once

#include <stdint.h>
#include <string>
#include <deque>
#include <tuple>

#include "models.h"

namespace ql {

    // Options for the optimization.
    struct Options {

        std::string data_root{"data"};
        int32_t batch_size{64};
        int32_t epochs{10};
        double lr{0.01};
        double momentum{0.5};
        bool cuda{true};
        int32_t seed{1};
        int32_t test_batch_size{1000};
        int32_t log_interval{10};
    };

    // Viable actions to take.
    enum Actions {

        UP = 0,
        DOWN = 1,
        LEFT = 2,
        RIGHT = 3,
        ROTATE_LEFT = 4,
        ROTATE_RIGHT = 5
    };

    using memory = std::deque<std::tuple<torch::Tensor /*state*/, 
                                        torch::Tensor /*action*/, 
                                        torch::Tensor /*reward*/, 
                                        torch::Tensor /*next_state*/>>;

    // Implements Q-Learning with variable model.
    class QLearning {

        public:

            QLearning(Model& policy_model, Model& target_model, 
                      Actions& action, Options& options);

            torch::Tensor Action(torch::Tensor& state);

            torch::Tensor Reward(torch::Tensor& state, torch::Tensor& action);

            void Push();

            void Optimize();

        private:

            // Replay memory.
            memory replay_memory_;

            // Sample from replay_memory_.
            memory Sample(uint32_t batch_size);
    };
} // namespace ql