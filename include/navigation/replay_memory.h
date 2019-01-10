#ifndef REPLAY_MEMORY_H_
#define REPLAY_MEMORY_H_

#include <deque>
#include <random>

#include <torch/torch.h>
#include <boost/ptr_container/ptr_vector.hpp>

using state = std::tuple<torch::Tensor /*left img state*/,
                         torch::Tensor /*right img state*/,
                         torch::Tensor /*action*/, 
                         torch::Tensor /*reward*/, 
                         torch::Tensor /*next_state*/>;

using memory = std::deque<state>;
using memoryptr = boost::ptr_vector<state>;

class ReplayMemory {

public:

    ReplayMemory(int64_t buffer_size, int64_t batch_size);

    void Add(state& state);

    memoryptr Sample();

    int64_t Length();

private:

    // Random engine.
    std::mt19937 random_engine_;

    // Memory.
    int64_t buffer_size_;
    int64_t batch_size_;
    memory memory_;

};

#endif