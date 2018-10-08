#! /usr/bin/env python
# -*- encoding: UTF-8 -*-


import rospy
from gem import GeM
from gem_tools import GeM_tools
import numpy as np
import cPickle as pickle
from std_msgs.msg import Int32 
from geometry_msgs.msg import WrenchStamped
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Imu




class  gem_ros():
	def __init__(self):
		rospy.init_node("gait_phase_estimation_module")

		left_foot_wrench_topic = rospy.get_param('gem_left_foot_wrench_topic')
		right_foot_wrench_topic = rospy.get_param('gem_right_foot_wrench_topic')
		imu_topic = rospy.get_param('gem_imu_topic')
		com_topic = rospy.get_param('gem_com_topic')
		freq = rospy.get_param('gem_freq')
		path = rospy.get_param('gem_train_path') 
	


		self.phase_pub = rospy.Publisher('/gem/phase', Int32, queue_size=1000)
		self.leg_pub = rospy.Publisher('/gem/leg', Int32, queue_size=1000)
		
		self.rft_sub  = rospy.Subscriber(right_foot_wrench_topic, WrenchStamped, self.lwrenchcb)
		self.lft_sub  = rospy.Subscriber(left_foot_wrench_topic, WrenchStamped, self.rwrenchcb)
		self.com_sub  = rospy.Subscriber(com_topic,PoseStamped,  self.comcb)
		self.com_sub  = rospy.Subscriber(imu_topic,Imu,  self.imucb)

		self.imu_inc = False
		self.lwrench_inc = False
		self.rwrench_inc = False
		self.com_inc = False
		self.freq = freq

		f = open(path+'/gem.save', 'rb')
		self.g = pickle.load(f)
		f.close()

		f = open(path+'/gem_tools.save', 'rb')
		self.gt = pickle.load(f)
		f.close()
	
		self.phase_msg = Int32()

		self.phase = -1	

		print('GEM Initialized')
	
	def lwrenchcb(self,msg):
		self.lwrench = msg
		self.lwrench_inc = True

	def rwrenchcb(self,msg):
		self.rwrench = msg
		self.rwrench_inc = True
	def comcb(self,msg):
		self.com = msg
		self.com_inc = True

	def imucb(self,msg):
		self.imu = msg
		self.imu_inc = True

	def predict(self):
		if(self.imu_inc and self.lwrench_inc and self.rwrench_inc and self.com_inc):
			self.imu_inc = False
			self.lwrench_inc = False
			self.rwrench_inc = False
			self.com_inc = False
			self.phase, self.reduced_data = self.g.predict(self.gt.genInputCF(self.com.pose.position.x,self.com.pose.position.y,self.com.pose.position.z,
self.imu.linear_acceleration.x,self.imu.linear_acceleration.y, self.imu.linear_acceleration.z, self.imu.angular_velocity.x, self.imu.angular_velocity.y, self.lwrench.wrench.force.x,self.lwrench.wrench.force.y,self.lwrench.wrench.force.z,self.rwrench.wrench.force.x,self.rwrench.wrench.force.y,self.rwrench.wrench.force.z, 
self.lwrench.wrench.torque.x,self.lwrench.wrench.torque.y,self.lwrench.wrench.torque.z,self.rwrench.wrench.torque.x,self.rwrench.wrench.torque.y,self.rwrench.wrench.torque.z))


	def run(self):
		r = rospy.Rate(self.freq) # 500hz
		while not rospy.is_shutdown():
			self.predict()
			
			self.phase_msg.data = self.phase        	
			self.phase_pub.publish(self.phase_msg)
			self.leg_pub.publish(self.phase_msg)
			r.sleep()
   		
if __name__ == "__main__":
    try:
        gr = gem_ros()
	gr.run()
    except rospy.ROSInterruptException:
	pass
