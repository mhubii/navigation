# Path Finding

This folder is currently under construction!

## Autonomous Navigation

### Behavioural Cloning






# Old Stuff to be Updated

## Building
### Build Dependencies
Install yarp as shown here http://www.yarp.it/install.html.
Install gazebo as shown here http://gazebosim.org/tutorials?tut=install_ubuntu.
Install gazebo-yarp-plugins as shown here http://robotology.gitlab.io/docs/gazebo-yarp-plugins/master/install.html.

Add the vehicle to the GAZEBO_MODEL_PATH environment variable. In your bashrc add the line:

```
export GAZEBO_MODEL_PATH=${GAZEBO_MODEL_PATH}:<location to path_finding>/path_finding
```
 
### Build Path Finding
Clone the repository:

```
git clone --recursive https://github.com/mhubii/path_finding.git
```

This will also clone the submodule ORB SLAM2. Build ORB SLAM2 as described here https://github.com/raulmur/ORB_SLAM2.git.
Next, build path finding:

```
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8
```

### Run an Example
Start the yarpserver. In a terminal tipe:

```
yarpserver --write
```

Open gazebo with root access so that the vehicle plugin can be used to steer the robot with W/A/S/D/LEFT/RIGHT. In another terminal type:

```
sudo -s
gazebo
```

Run ORB SLAM2 in a new terminal:

```
cd bin
./orb_slam
```
