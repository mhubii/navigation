#ifndef RL_AGENT_H_
#define RL_AGENT_H_

#include "py_tensor.h"

// Default name of the Python module to load.
#define DEFAULT_RL_MODULE "dqn_agent"

// Default name of the Python function from the user's module
// which infers the next action from the current state.
// The expected function is of the form 'def next_action(state):',
// where state is a PyTorch tensor containing the environment,
// and the function return the predicted action.
#define DEFAULT_NEXT_ACTION "next_action"

// Default name of the Python function from the user's module
// which recieves rewards and performs training.
// The expected reward function is of the form 'def next_reward(state, reward, new_episode):',
// where the function returns the predicted action and accepts the reward.
#define DEFAULT_NEXT_REWARD "next_reward"

// Default name of the Python function for loading model checkpoints.
#define DEFAULT_LOAD_MODEL "load_model"

// Default name of the Python function for saving model checkpoints.
#define DEFAULT_SAVE_MODEL "save_model"

// Base class for deep reinforcement learning agent,
// using Python & PyTorch underneath with C FFI.
class RLAgent
{
public:

	// Create a new instance of a module for training an agent.
	static RLAgent* Create();

	// Destructor.
	virtual ~RLAgent();

	// From the input state, predict the next action.
	virtual bool NextAction();

	// Issue the next reward and training iteration.
	virtual bool NextReward();

	// Load model checkpoint.
	virtual bool LoadCheckpoint();

	// Save model checkpoint.
	virtual bool SaveCheckpoint();

	// Globally load Python scripting interpreter.
	static bool LoadInterpreter();

	// Load Python script module.
	bool LoadModule();

protected:

	RLAgent();

	virtual bool Init();

	uint32_t input_width_;
	uint32_t input_height_;
	uint32_t num_inputs_;
	uint32_t num_actions_;

	Tensor* reward_tensor_;
	Tensor* action_tensor_;

	enum {
	
		ACTION_FUNCTION = 0,
		REWARD_FUNCTION,
		LOAD_FUNCTION,
		SAVE_FUNCTION,
		NUM_FUNCTIONS
	};

	std::string module_name_;
	void* module_obj_;
	void* function[NUM_FUNCTIONS];
	std::string function_name_[NUM_FUNCTIONS];

	static bool scripting_loaded_;
};

#endif
