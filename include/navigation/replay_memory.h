#ifndef REPLAY_MEMORY_H_
#define REPLAY_MEMORY_H_

#include <deque>
#include <random>

#include <torch/torch.h>
#include <Eigen/Core>

// States.
using state = std::tuple<torch::Tensor /*left img state*/,
                         torch::Tensor /*right img state*/,
                         torch::Tensor /*action*/, 
                         torch::Tensor /*reward*/, 
                         torch::Tensor /*left img next_state*/,
                         torch::Tensor /*right img next state*/,
                         torch::Tensor /*termination state*/>;

// All states in the buffer.
using memory = std::deque<state>;

struct states_batch {

    // Same as state but for a whole batch.
    torch::Tensor left_imgs;
    torch::Tensor right_imgs;
    torch::Tensor actions;
    torch::Tensor rewards;
    torch::Tensor next_left_imgs;
    torch::Tensor next_right_imgs;
    torch::Tensor dones;

    // Operator to check for uninitialized structs.
	bool operator !() const {

		return (this->left_imgs.numel()       == 0 || 
                this->right_imgs.numel()      == 0 ||
                this->actions.numel()         == 0 ||
                this->rewards.numel()         == 0 ||
                this->next_left_imgs.numel()  == 0 ||
                this->next_right_imgs.numel() == 0 ||
                this->dones.numel()           == 0);
	};

    explicit operator bool() const {

        return (this->left_imgs.numel()       != 0 && 
                this->right_imgs.numel()      != 0 &&
                this->actions.numel()         != 0 &&
                this->rewards.numel()         != 0 &&
                this->next_left_imgs.numel()  != 0 &&
                this->next_right_imgs.numel() != 0 &&
                this->dones.numel()           != 0);
    }
};


class ReplayMemory {

public:

    ReplayMemory(int64_t buffer_size, int64_t batch_size);

    void Add(state& state);

    states_batch Sample(torch::Device device);

    int64_t Length();

private:

    // Random engine.
    std::random_device rd_;
    std::mt19937 random_engine_;

    // Memory.
    int64_t buffer_size_;
    int64_t batch_size_;
    memory deque_;
};

#endif