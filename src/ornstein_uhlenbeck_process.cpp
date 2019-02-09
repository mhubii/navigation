#include "ornstein_uhlenbeck_process.h"

OUProcess::OUProcess(torch::IntList size, float mu, float theta, float sigma) 
    : mu_(mu*torch::ones(size)),
      state_(mu*torch::ones(size)),
      theta_(theta),
      sigma_(sigma) {

};

void OUProcess::Reset() {

    // Reset the noise to mean.
    state_ = mu_;
};

torch::Tensor OUProcess::Sample() {

    // Update internal state and return it as noise sample.
    torch::Tensor x = state_;
    torch::Tensor dx = theta_*(mu_ - x) + sigma_*torch::rand(state_.sizes());

    state_ = x + dx;

    return state_;
};