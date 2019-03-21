# gem
Gait-phase Estimation Module (GEM) for Humanoid Robot Walking. The code is open-source (BSD License). Please note that this work is an on-going research and thus some parts are not fully developed yet. Furthermore, the code will be subject to changes in the future which could include greater re-factoring.

GEM is an unsupervised learning framework which employs a 2D latent space and Gaussian Mixture Models (GMMs) to facilitate accurate prediction/classification of the gait phase during locomotion.

Nevertheless, GEM can be used for real-time gait phase estimation without training based on the contact wrenches and optionally kinematics. The latter functionality facilitates the case where not enough training data can be obtained.


Video: https://www.youtube.com/watch?v=w09yb81IXpQ

Papers: 
* Unsupervised Gait Phase Estimation for Humanoid Robot Walking (to appear at Intl. Conf. on Robotics and Automation (ICRA), 2019)

## Training
Solely proprioceptive sensing is utilized in training, namely joint encoder, F/T, and IMU.


<p align="center">
  <img width="701" height="693" src="img/gem01.png">
</p>


## Real-time Gait-Phase Prediction
GEM can be readily employed in real-time for estimating the gait phase. The latter is accomplished by either loading a trained GEM python module and use it for real-time preditiction or by utilizying GEM for real-time estimation based on the sensed contact wrenches and optionally leg kinematics.

<p align="center">
  <img width="708" height="393" src="img/gem02.png">
</p>



## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites
* Ubuntu 14.04 and later
* ROS indigo and later
* Sklearn 
* Keras 
* tensorflow
* tested on python3 (3.6.5) and python (2.7.2)

## Installing
* pip install sklearn
* pip install keras
* pip install tensorflow
* git clone https://github.com/mrsp/gem.git
* catkin_make
* If you are using catkin tools run: catkin build  

## ROS Examples
### Train the Valkyrie module
* Download the valkyrie bag file from [valk_bagfile](http://users.ics.forth.gr/~spiperakis/gem_test_valkyrie.zip)
* Uncompress
* train: python train.py 

### Train your own module
* Save the corresponding files in a similar form as the valkyrie bag file from [valk_bagfile](http://users.ics.forth.gr/~spiperakis/gem_test_valkyrie.zip)
* train: python train.py

### Run in real-time to infer the gait-phase:
* configure appropriately the config yaml file (in config folder) with the corresponding topics 
* roslaunch gem gem_ros.launch
