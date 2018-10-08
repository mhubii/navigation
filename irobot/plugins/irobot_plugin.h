#ifndef GAZEBO_IROBOT_PLUGIN_H_
#define GAZEBO_IROBOT_PLUGIN_H_

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <ignition/math.hh>

#include <stdio.h>
#include <vector>

#include "dqn_agent.h"
#include "read_cams.h"
#include "keyboard.h"

namespace gazebo
{

class IRobotPlugin : public ModelPlugin
{
public:
	IRobotPlugin();

	void Load(physics::ModelPtr parent, sdf::ElementPtr /*sdf*/);	
	
	void OnUpdate();

	bool CreateAgent();

	static const uint32_t DOF = 3; // fwd/back, left/right, rotation_left/rotation_right

private:

	// Joint velocity control.
	double vel_[DOF];

	// Operating mode.
	enum OperatingMode {
		USER_MANUAL
	} op_mode_;

	// Initialize joints.
	bool ConfigureJoints(const char* name);

	// Update joints.
	bool UpdateJoints();

	// Members.
	physics::ModelPtr model;

	event::ConnectionPtr update_connection;	

	std::vector<physics::JointPtr> joints_;

	// Incremental velocity change.
	double vel_delta_;

	// Keyboard for manual operating mode.
	Keyboard* keyboard_;

	// Reinforcement learning agent.
	RLAgent* agent_;
};

// Register the plugin.
GZ_REGISTER_MODEL_PLUGIN(IRobotPlugin)
}

#endif
