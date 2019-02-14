#include "replay_memory.h"

ReplayMemory::ReplayMemory(int64_t buffer_size, int64_t batch_size)
    : rd_(),
      random_engine_(rd_()),
      buffer_size_(buffer_size),
      batch_size_(batch_size) {

}

void ReplayMemory::Add(state& state) {

    // Add state to memory.
    if (deque_.size() == buffer_size_) {

        deque_.pop_front();
        deque_.push_back(state);
    }
    else {

        deque_.push_back(state);
    }
}

states_batch ReplayMemory::Sample(torch::Device device) {

    // Generate random indices.
    std::vector<int> idx;
    idx.reserve(deque_.size());

    for (uint i = 0; i < batch_size_; i++) {

        idx.push_back(std::uniform_int_distribution<int>(0, deque_.size() - 1)(random_engine_));
    }

	// Sample from deque.
	if (deque_.size() > batch_size_) {

        // Get the shapes. There are some issues with torch::IntList, thats why its done everytime.
        torch::IntList img_shape = std::get<0>(deque_[0]).sizes();
        torch::IntList action_shape = std::get<2>(deque_[0]).sizes();

        // Allocate empty tensors.
		torch::Tensor left_imgs = torch::zeros({batch_size_, img_shape[1], img_shape[2], img_shape[3]}, std::get<0>(deque_[0]).type());
		torch::Tensor right_imgs = torch::zeros({batch_size_, img_shape[1], img_shape[2], img_shape[3]}, std::get<1>(deque_[0]).type());
        torch::Tensor actions = torch::zeros({batch_size_, action_shape[1]}, std::get<2>(deque_[0]).type());
        torch::Tensor rewards = torch::zeros({batch_size_, 1}, std::get<3>(deque_[0]).type());
        torch::Tensor next_left_imgs = torch::zeros({batch_size_, img_shape[1], img_shape[2], img_shape[3]}, std::get<4>(deque_[0]).type());
        torch::Tensor next_right_imgs = torch::zeros({batch_size_, img_shape[1], img_shape[2], img_shape[3]}, std::get<5>(deque_[0]).type());
        torch::Tensor dones = torch::zeros({batch_size_, 1}, std::get<6>(deque_[0]).type());

		for (int i = 0; i < batch_size_; i++) {

			left_imgs.slice(0, i, i+1) = std::get<0>(deque_[idx[i]]);
			right_imgs.slice(0, i, i+1) = std::get<1>(deque_[idx[i]]);
            actions.slice(0, i, i+1) = std::get<2>(deque_[idx[i]]);
            rewards.slice(0, i, i+1) = std::get<3>(deque_[idx[i]]);
            next_left_imgs.slice(0, i, i+1) = std::get<4>(deque_[idx[i]]);
            next_right_imgs.slice(0, i, i+1) = std::get<5>(deque_[idx[i]]);
            dones.slice(0, i, i+1) = std::get<6>(deque_[idx[i]]);
		}

		return {left_imgs.to(device), right_imgs.to(device), actions.to(device), rewards.to(device), next_left_imgs.to(device), next_right_imgs.to(device), dones.to(device)};
	}

	return {};
}

int64_t ReplayMemory::Length() {

    return deque_.size();
}