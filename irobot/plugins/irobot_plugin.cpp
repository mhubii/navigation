#include "irobot_plugin.h"

#define L_FRONT_PITCH "irobot::l_front_wheel_pitch"
#define L_FRONT_ROLL "irobot::l_front_wheel_roll"
#define R_FRONT_PITCH "irobot::r_front_wheel_pitch"
#define R_FRONT_ROLL "irobot::r_front_wheel_roll"
#define L_BACK_PITCH "irobot::l_back_wheel_pitch"
#define L_BACK_ROLL "irobot::l_back_wheel_roll"
#define R_BACK_PITCH "irobot::r_back_wheel_pitch"
#define R_BACK_ROLL "irobot::r_back_wheel_roll"

#define VELOCITY_MIN -1.0f
#define VELOCITY_MAX  1.0f

namespace gazebo
{

IRobotPlugin::IRobotPlugin() :
	ModelPlugin() {

	op_mode_   = USER_MANUAL;
	vel_delta_ = 1e-3;

	for (int i = 0; i < DOF; i++) {
	
		vel_[i] = 0.;
	}

	keyboard_ = Keyboard::Create();
}

void IRobotPlugin::Load(physics::ModelPtr parent, sdf::ElementPtr /*sdf*/) {
	
	// Store the pointer to the model.
	this->model = parent;

	// Configure the joints.
	ConfigureJoints(L_FRONT_PITCH);
	ConfigureJoints(R_FRONT_PITCH);
	ConfigureJoints(L_BACK_PITCH);
	ConfigureJoints(R_BACK_PITCH);

	ConfigureJoints(L_FRONT_ROLL);
	ConfigureJoints(R_FRONT_ROLL);
	ConfigureJoints(L_BACK_ROLL);
	ConfigureJoints(R_BACK_ROLL);

	// Listen to the update event. This event is broadcast every simulation iterartion.
	this->update_connection = event::Events::ConnectWorldUpdateBegin(std::bind(&IRobotPlugin::OnUpdate, this));
}

void IRobotPlugin::OnUpdate() {

	if (UpdateJoints()) {

		for(int i = 0; i < DOF; i++) {
			if(vel_[i] < VELOCITY_MIN)
				vel_[i] = VELOCITY_MIN;

			if(vel_[i] > VELOCITY_MAX)
				vel_[i] = VELOCITY_MAX;
		}

		if (joints_.size() != 8) {
			
			printf("IRobotPlugin -- could only find %zu of 8 drive joints\n", joints_.size());
			return;
		}

		// Drive forward/backward and turn.
		joints_[0]->SetVelocity(0, vel_[0]); // left
		joints_[1]->SetVelocity(0, vel_[1]); // right
		joints_[2]->SetVelocity(0, vel_[0]); // left
		joints_[3]->SetVelocity(0, vel_[1]); // right

		// Drive left/right. Rotate to frames.
		ignition::math::Vector3<double> axis = axis.UnitX;
		ignition::math::Vector3<double> tmp = tmp.Zero;

		ignition::math::Quaterniond ori = joints_[0]->AxisFrameOffset(0);
		tmp = ori.RotateVector(axis);
		joints_[4]->SetAxis(0, tmp);
		joints_[4]->SetVelocity(0, vel_[2]);

		ori = joints_[1]->AxisFrameOffset(0);		
		tmp = ori.RotateVector(axis);
		joints_[5]->SetAxis(0, tmp);
		joints_[5]->SetVelocity(0, vel_[2]);

		ori = joints_[2]->AxisFrameOffset(0);
		tmp = ori.RotateVector(axis);
		joints_[6]->SetAxis(0, tmp);
		joints_[6]->SetVelocity(0, vel_[2]);

		ori = joints_[3]->AxisFrameOffset(0);
		tmp = ori.RotateVector(axis);
		joints_[7]->SetAxis(0, tmp);
		joints_[7]->SetVelocity(0, vel_[2]);
	}
}

bool IRobotPlugin::ConfigureJoints(const char* name) {

	std::vector<physics::JointPtr> joints = model->GetJoints();
	const size_t num_joints = joints.size();

	for (int i = 0; i < num_joints; i++) {

		if (strcmp(name, joints[i]->GetScopedName().c_str()) == 0) {
			
			joints[i]->SetVelocity(0, 0);
			joints_.push_back(joints[i]);
			return true;
		}
	}

	printf("IRobotPlugin -- failed to find joint '%s'\n", name);
	return false;
}

bool IRobotPlugin::UpdateJoints() {

	if (op_mode_ == USER_MANUAL) {

		keyboard_->Poll();
		
		if (keyboard_->KeyDown(KEY_W)) {
			
			vel_[0] += vel_delta_;
			vel_[1] += vel_delta_;
		}
		if (keyboard_->KeyDown(KEY_S)) {
			
			vel_[0] -= vel_delta_;
			vel_[1] -= vel_delta_;
		}
		if (keyboard_->KeyDown(KEY_D)) {
			
			vel_[0] += vel_delta_;
			vel_[1] -= vel_delta_;
		}
		if (keyboard_->KeyDown(KEY_A)) {
			
			vel_[0] -= vel_delta_;
			vel_[1] += vel_delta_;
		}
		if (keyboard_->KeyDown(KEY_LEFT)) {
			
			vel_[2] -= vel_delta_;
		}
		if (keyboard_->KeyDown(KEY_RIGHT)) {
			
			vel_[2] += vel_delta_;
		}
		if (keyboard_->KeyDown(KEY_E)) {
			
			for (int i = 0; i < DOF; i++) {
	
				vel_[i] = 0.;
			}
		}

	}

	return true;
}
} // End of namespace gazebo.
