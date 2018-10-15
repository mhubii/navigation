#include "q_learning.h"

namespace caffe2 {

QLearning::QLearning(ModelBase& model, int replay_memory_capacity, double gamma) 
    : replay_memory_capacity_(replay_memory_capacity),
      gamma_(gamma) {

}

void QLearning::Initialize() {


}
} // End of namespace dqn.