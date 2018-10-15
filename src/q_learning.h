#ifndef DQN_H_
#define DQN_H_

#include <deque>
#include <opencv2/opencv.hpp>

#include "model_base.h"
#include "action.h"

namespace caffe2{

class QLearning
{
public:

	QLearning(ModelBase& model, int replay_memory_capacity, double gamma);

	void Initialize();

	Action SelectAction();
	
	void AddTransition();

	void Update();

	inline const int& GetMemorySize() const { return replay_memory_.size(); };

	inline const int& GetCurrentIter() const { return current_iter_; };

private:

	// const std::string solver_param_;
	const int replay_memory_capacity_;
	const double gamma_;
	int current_iter_;

	std::deque<std::tuple<cv::Mat, Action, float>> replay_memory_;
};

} // End of namespace dqn.
#endif
