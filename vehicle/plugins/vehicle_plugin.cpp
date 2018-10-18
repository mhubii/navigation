#include "vehicle_plugin.h"

#define L_FRONT_PITCH "vehicle::l_front_wheel_pitch"
#define L_FRONT_ROLL "vehicle::l_front_wheel_roll"
#define R_FRONT_PITCH "vehicle::r_front_wheel_pitch"
#define R_FRONT_ROLL "vehicle::r_front_wheel_roll"
#define L_BACK_PITCH "vehicle::l_back_wheel_pitch"
#define L_BACK_ROLL "vehicle::l_back_wheel_roll"
#define R_BACK_PITCH "vehicle::r_back_wheel_pitch"
#define R_BACK_ROLL "vehicle::r_back_wheel_roll"

#define VELOCITY_MIN -10.0f
#define VELOCITY_MAX  10.0f

#define WORLD_NAME "vehicle_world"
#define VEHICLE_NAME "vehicle"
#define GOAL_NAME "goal"

namespace gazebo
{

VehiclePlugin::VehiclePlugin() :
	ModelPlugin() {
printf("hay2\n");
	op_mode_   = USER_MANUAL;
	//new_state_ = false;
	vel_delta_ = 1e-3;
printf("hay6\n");
	for (int i = 0; i < DOF; i++) {
	
		vel_[i] = 0.;
	}
printf("hay7\n");
	keyboard_ = Keyboard::Create();
	printf("hay8\n");
}

void VehiclePlugin::Load(physics::ModelPtr parent, sdf::ElementPtr /*sdf*/) {
	printf("hay9\n");
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

	// Create Q-Learning agent.
	//if (!CreateAgent()) {

	//	printf("VehiclePlugin -- failed to create Q-Learning agent\n");
	//}

	// Create a node for camera communication.
	//multi_camera_node_->Init();
	//multi_camera_sub_ = multi_camera_node_->Subscribe("/gazebo/" WORLD_NAME "/" VEHICLE_NAME "/chassis/stereo_camera/images", &VehiclePlugin::OnCameraMsg, this);

	// Create a node for collision detection.

	// Listen to the update event. This event is broadcast every simulation iterartion.
	this->update_connection = event::Events::ConnectWorldUpdateBegin(std::bind(&VehiclePlugin::OnUpdate, this));
}

void VehiclePlugin::OnUpdate() {
printf("hay3\n");
	if (UpdateJoints()) {

		for(int i = 0; i < DOF; i++) {
			if(vel_[i] < VELOCITY_MIN)
				vel_[i] = VELOCITY_MIN;

			if(vel_[i] > VELOCITY_MAX)
				vel_[i] = VELOCITY_MAX;
		}

		if (joints_.size() != 8) {
			
			printf("VehiclePlugin -- could only find %zu of 8 drive joints\n", joints_.size());
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
/*
void VehiclePlugin::OnCameraMsg(ConstImageStampedPtr &msg) {

	if (!msg) {

		printf("VehiclePlugin -- received NULL message\n");
		return;
	}

	const int width = msg->image().width();
	const int height = msg->image().height();
	const int bpp = (msg->image().step()/msg->image().width())*8; // Bits per pixel.
	const int size = msg->image().data().size();

	if (bpp != 24) {

		printf("VehiclePlugin -- expected 24 bits per pixel uchar3 image from camera, got %i\n", bpp);
		return;
	}

	// 
	printf("Got an image of size: %ix%i", height, width);
	
	new_state_ = true;
}

void VehiclePlugin::OnCollisionMsg(ConstContactsPtr &contacts) {

}

bool VehiclePlugin::CreateAgent() {

	return true;
}


bool VehiclePlugin::UpdateAgent() {

	return true;
}
*/
bool VehiclePlugin::ConfigureJoints(const char* name) {
printf("hay4\n");
	std::vector<physics::JointPtr> joints = model->GetJoints();
	const size_t num_joints = joints.size();

	for (int i = 0; i < num_joints; i++) {

		if (strcmp(name, joints[i]->GetScopedName().c_str()) == 0) {
			
			joints[i]->SetVelocity(0, 0);
			joints_.push_back(joints[i]);
			return true;
		}
	}

	printf("VehiclePlugin -- failed to find joint '%s'\n", name);
	return false;
}

bool VehiclePlugin::UpdateJoints() {
printf("hay5\n");
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

		return true;
	}
	else if (op_mode_ == AUTONOMOUSLY) {


		// No new processed state.
		//new_state_ = false;

		//if (UpdateAgent()) {
			
		//	return true;
		//}
	}

	return false;
}
} // End of namespace gazebo.
