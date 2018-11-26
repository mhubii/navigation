#include "q_learning.h"

namespace ql {

    QLearning::QLearning(Model& policy_model, Model& target_model, 
                         Actions& action, Options& options) {

    }

    torch::Tensor QLearning::Action(torch::Tensor& state) {

    }

    torch::Tensor QLearning::Reward(torch::Tensor& state, torch::Tensor& action) {

    }

    void QLearning::Push() {

    }

    void QLearning::Optimize() {

    }

    memory QLearning::Sample(uint32_t batch_size) {

    }


}
