#include "replay_memory.h"

ReplayMemory::ReplayMemory(int64_t buffer_size, int64_t batch_size)
    : random_engine_(0),
      initialized_(false),
      buffer_size_(buffer_size),
      batch_size_(batch_size),
      img_shape_(0),
      action_shape_(0) {

}

void ReplayMemory::Add(state& state) {

    if (!initialized_) {

        // Get shapes if not initialized.
        img_shape_ = std::get<0>(state).sizes();
        action_shape_ = std::get<2>(state).sizes();

        initialized_ = true;
    };

    // Add state to memory.
    if (deque_.size() == buffer_size_) {
    
        deque_.pop_front();
    }

    deque_.push_back(state);
}

states_batch ReplayMemory::Sample() {

    // Generate random indices.
    std::vector<int> idx;
    idx.reserve(deque_.size());

    for (uint i = 0; i < batch_size_; i++) {

        idx.push_back(std::uniform_int_distribution<int>(0, deque_.size() - 1)(random_engine_));
    }

	// Sample from deque.
	if (deque_.size() > batch_size_) {

		torch::Tensor left_imgs = torch::empty({batch_size_, img_shape_[1], img_shape_[2], img_shape_[3]});
		torch::Tensor right_imgs = torch::empty({batch_size_, img_shape_[1], img_shape_[2], img_shape_[3]});
        torch::Tensor actions = torch::empty({batch_size_, action_shape_[1]});
        torch::Tensor rewards = torch::empty({batch_size_, 1});
        torch::Tensor next_left_imgs = torch::empty({batch_size_, img_shape_[1], img_shape_[2], img_shape_[3]});
        torch::Tensor next_right_imgs = torch::empty({batch_size_, img_shape_[1], img_shape_[2], img_shape_[3]});
        torch::Tensor dones = torch::empty({batch_size_, 1});

		for (int i = 0; i < batch_size_; i++) {

			left_imgs.slice(0, i, i+1) = std::get<0>(deque_[idx[i]]);
			right_imgs.slice(0, i, i+1) = std::get<1>(deque_[idx[i]]);
            actions.slice(0, i, i+1) = std::get<2>(deque_[idx[i]]);
            rewards.slice(0, i, i+1) = std::get<3>(deque_[idx[i]]);
            next_left_imgs.slice(0, i, i+1) = std::get<4>(deque_[idx[i]]);
            next_right_imgs.slice(0, i, i+1) = std::get<5>(deque_[idx[i]]);
            dones.slice(0, i, i+1) = std::get<6>(deque_[idx[i]]);
		}

		return {left_imgs, right_imgs, actions, rewards, next_left_imgs, next_right_imgs, dones};
	}

	return {};
}

int64_t ReplayMemory::Length() {

    return deque_.size();
}