#! /usr/bin/env python
# -*- encoding: UTF-8 -*-


'''
 * GeM - Gait-phase Estimation Module
 *
 * Copyright 2018-2019 Stylianos Piperakis, Foundation for Research and Technology Hellas (FORTH)
 * License: BSD
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Foundation for Research and Technology Hellas (FORTH) 
 *	 nor the names of its contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
'''


import rospy
from gem import GeM
from gem_tools import GeM_tools
import numpy as np
import pickle
from std_msgs.msg import Int32 
from std_msgs.msg import String 
from geometry_msgs.msg import WrenchStamped
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Imu
from gem import Gaussian 


class  gem_ros():
	def __init__(self):
		rospy.init_node("gait_phase_estimation_module")
		left_foot_wrench_topic = rospy.get_param('gem_left_foot_wrench_topic')
		right_foot_wrench_topic = rospy.get_param('gem_right_foot_wrench_topic')
		self.phase_pub = rospy.Publisher('/gem/phase', Int32, queue_size=1000)
		self.leg_pub = rospy.Publisher('/gem/leg', String, queue_size=1000)
		self.rft_sub  = rospy.Subscriber(right_foot_wrench_topic, WrenchStamped, self.lwrenchcb)
		self.lft_sub  = rospy.Subscriber(left_foot_wrench_topic, WrenchStamped, self.rwrenchcb)
		self.lwrench_inc = False
		self.rwrench_inc = False
		freq = rospy.get_param('gem_freq')
		self.freq = freq
		self.phase_msg = Int32()
		self.phase = -1	
		self.support_msg = String()

		useUL = rospy.get_param('useUL')

		if(useUL):
			imu_topic = rospy.get_param('gem_imu_topic')
			com_topic = rospy.get_param('gem_com_topic')
			path = rospy.get_param('gem_train_path') 	
			self.com_sub  = rospy.Subscriber(com_topic,PoseStamped,  self.comcb)
			self.com_sub  = rospy.Subscriber(imu_topic,Imu,  self.imucb)
			self.imu_inc = False
			self.com_inc = False
			f = open(path+'/gem.save', 'rb')
			self.g = pickle.load(f)
			f.close()
			f = open(path+'/gem_tools.save', 'rb')
			self.gt = pickle.load(f)
			f.close()
		else:
			self.g = GeM()
			g.setFrames(rospy.get_param('gem_lfoot_frame'), rospy.get_param('gem_rfoot_frame'))
			self.xmax = rospy.get_param('gem_foot_xmax')
			self.xmin = rospy.get_param('gem_foot_xmin')
			self.ymax = rospy.get_param('gem_foot_ymax')
			self.ymin = rospy.get_param('gem_foot_ymin')
			self.lfmin = rospy.get_param('gem_lfoot_force_thres')
			self.rfmin = rospy.get_param('gem_rfoot_force_thres')
			self.sigmalf = rospy.get_param('gem_lforce_std')
			self.sigmarf = rospy.get_param('gem_rforce_std')
			self.sigmalc = rospy.get_param('gem_lcop_std')
			self.sigmarc = rospy.get_param('gem_rcop_std')
		print('Gait-Phase Estimation Module Initialized Successfully')
	
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

	def predictUL(self):
		if(self.imu_inc and self.lwrench_inc and self.rwrench_inc and self.com_inc):
			self.imu_inc = False
			self.lwrench_inc = False
			self.rwrench_inc = False
			self.com_inc = False
			self.phase, self.reduced_data = self.g.predict(self.gt.genInputCF(self.com.pose.position.x,self.com.pose.position.y,self.com.pose.position.z,
self.imu.linear_acceleration.x,self.imu.linear_acceleration.y, self.imu.linear_acceleration.z, self.imu.angular_velocity.x, self.imu.angular_velocity.y, self.lwrench.wrench.force.x,self.lwrench.wrench.force.y,self.lwrench.wrench.force.z,self.rwrench.wrench.force.x,self.rwrench.wrench.force.y,self.rwrench.wrench.force.z, 
self.lwrench.wrench.torque.x,self.lwrench.wrench.torque.y,self.lwrench.wrench.torque.z,self.rwrench.wrench.torque.x,self.rwrench.wrench.torque.y,self.rwrench.wrench.torque.z))
			self.support_leg = self.g.getSupportLeg()


	def predictFT(self):
		if(self.lwrench_inc and self.rwrench_inc):
			self.lwrench_inc = False
			self.rwrench_inc = False
			lfz = self.lwrench.wrench.force.z
			rfz = self.rwrench.wrench.force.z
			ltx = self.lwrench.wrench.torque.x
			lty = self.lwrench.wrench.torque.y
			rtx = self.rwrench.wrench.torque.x
			rty = self.rwrench.wrench.torque.y
					

			if(lfz>0):
				coplx = -lty/lfz
				coply = ltx/lfz
			else:
				coplx = 0
				coply = 0

			if(rfz>0):
				coprx = -rty/rfz
				copry = rtx/rfz
			else:
				coprx = 0
				copry = 0

			self.phase = self.predictFT(lfz,  rfz,  coplx,  coply,  coprx,  copry, self.xmax,
			self.xmin, self.ymax, self.ymin, self.lfmin, self.rfmin, self.sigmalf, self.sigmarf, self.sigmalc, self.sigmarc)
			self.support_leg = self.g.getSupportLeg()


	def run(self):
		r = rospy.Rate(self.freq) 
		while not rospy.is_shutdown():
			if(useUL):
				self.predictUL()
			else:
				self.predictFT()
			self.phase_msg.data = self.phase        	
			self.phase_pub.publish(self.phase_msg)
			self.support_msg.data = self.support_leg        	
			self.leg_pub.publish(self.support_msg)
			r.sleep()
   		
if __name__ == "__main__":
	gr = gem_ros()
    try:
		gr.run()
    except rospy.ROSInterruptException:
		pass