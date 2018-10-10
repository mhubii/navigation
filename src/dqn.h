#ifndef DQN_H_
#define DQN_H_

namespace dqn{

enum Action {

	UP = 0,
	DOWN = 1,
	LEFT = 2,
	RIGHT = 3,
	ROTATE_LEFT = 4,
	ROTATE_RIGHT = 5
};

class DQN
{
public:
	DQN();

	void Initialize();

	void LoadTrainedModel();

	Action SelectAction();
	
	void AddTransition();

	void Update();

	//inline const int& GetMemorySize() const { return replay_memory_.size(); };

	inline const int& GetCurrentIter() const { return current_iter_; };

private:

	//std::deque<std::tuple<some kind of image, Action, float>> replay_memory_

	int current_iter_;
	

};

} // End of namespace dqn.
#endif
