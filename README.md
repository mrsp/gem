# gem
Gait-phase Estimation Module (GEM) for Humanoid Robot Walking. The code is open-source (BSD License). Please note that this work is an on-going research and thus some parts are not fully developed yet. Furthermore, the code will be subject to changes in the future which could include greater re-factoring.



## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites
* Ubuntu 14.04 and later
* ROS indigo and later
* Sklearn 
* Keras 

## Installing
* pip install sklearn
* pip install keras
* git clone https://github.com/mrsp/gem.git
* catkin_make -DCMAKE_BUILD_TYPE=Release 
* If you are using catkin tools run: catkin build  --cmake-args -DCMAKE_BUILD_TYPE=Release 

## ROS Examples
### Valkyrie SRCsim
* Download the trained module from [valk_train](http://users.ics.forth.gr/~spiperakis/gem.save) and the trained tools from [valk_train_tools](http://users.ics.forth.gr/~spiperakis/gem_tools.save)
* Create a folder train with the saved files.
* (SIMULATOR - Valkyrie with Gazebo) launch [srcsim](https://bitbucket.org/osrf/srcsim) or download the bag file [valk_bag](http://users.ics.forth.gr/~spiperakis/gem_valk.bag)
* roscore 
* rosbag play gem_valk.bag
* roslaunch gem gem_ros.launch



### Train your own module
* Download the valkyrie bag file from [valk_bagfile](http://users.ics.forth.gr/~spiperakis/gem_test_valkyrie.zip)
* Uncompress

