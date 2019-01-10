#include "replay_memory.h"

ReplayMemory::ReplayMemory(int64_t buffer_size, int64_t batch_size)
    : random_engine_(0),
      buffer_size_(buffer_size),
      batch_size_(batch_size) {

}

void ReplayMemory::Add(state& state) {

    // Add state to memory.
    if (memory_.size() == buffer_size_) {
    
        memory_.pop_front();
    }

    memory_.push_back(state);
}

memoryptr ReplayMemory::Sample() {

    // Generate random indices.
    std::vector<int> idx;
    idx.reserve(batch_size_);

    for (uint i = 0; i < batch_size_; i++) {

        idx.push_back(std::uniform_int_distribution<int>(0, buffer_size_ - 1)(random_engine_));
    }

    memoryptr mem_ptr;

    // Sample from memory.
    for (uint i = 0; i < batch_size_; i++) {

        mem_ptr.push_back(&memory_[idx[i]]);
    }

    return mem_ptr;
}

int64_t ReplayMemory::Length() {

    return memory_.size();
}