/*
 * SERoW - a complete state estimation scheme for humanoid robots
 *
 * Copyright 2017-2018 Stylianos Piperakis, Foundation for Research and Technology Hellas (FORTH)
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
 */

#include <iostream>
#include <algorithm>
#include <gem/robotDyn.h>

void humanoid_state_publisher::loadparams()
{
    ros::NodeHandle n_p("~");
    // Load Server Parameters
    n_p.param<std::string>("modelname", modelname, "nao.urdf");
    rd = new serow::robotDyn(modelname, false);
    n_p.param<std::string>("base_link", base_link_frame, "base_link");
    n_p.param<std::string>("lfoot", lfoot_frame, "l_ankle");
    n_p.param<std::string>("rfoot", rfoot_frame, "r_ankle");
    n_p.param<double>("imu_topic_freq", freq, 100.0);
    n_p.param<double>("fsr_topic_freq", fsr_freq, 100.0);
    n_p.param<double>("joint_topic_freq", joint_freq, 100.0);
    n_p.param<double>("joint_cutoff_freq", joint_cutoff_freq, 10.0);

    n_p.getParam("T_B_A", affine_list);
    T_B_A(0, 0) = affine_list[0];
    T_B_A(0, 1) = affine_list[1];
    T_B_A(0, 2) = affine_list[2];
    T_B_A(0, 3) = affine_list[3];
    T_B_A(1, 0) = affine_list[4];
    T_B_A(1, 1) = affine_list[5];
    T_B_A(1, 2) = affine_list[6];
    T_B_A(1, 3) = affine_list[7];
    T_B_A(2, 0) = affine_list[8];
    T_B_A(2, 1) = affine_list[9];
    T_B_A(2, 2) = affine_list[10];
    T_B_A(2, 3) = affine_list[11];
    T_B_A(3, 0) = affine_list[12];
    T_B_A(3, 1) = affine_list[13];
    T_B_A(3, 2) = affine_list[14];
    T_B_A(3, 3) = affine_list[15];

    n_p.getParam("T_B_G", affine_list);
    T_B_G(0, 0) = affine_list[0];
    T_B_G(0, 1) = affine_list[1];
    T_B_G(0, 2) = affine_list[2];
    T_B_G(0, 3) = affine_list[3];
    T_B_G(1, 0) = affine_list[4];
    T_B_G(1, 1) = affine_list[5];
    T_B_G(1, 2) = affine_list[6];
    T_B_G(1, 3) = affine_list[7];
    T_B_G(2, 0) = affine_list[8];
    T_B_G(2, 1) = affine_list[9];
    T_B_G(2, 2) = affine_list[10];
    T_B_G(2, 3) = affine_list[11];
    T_B_G(3, 0) = affine_list[12];
    T_B_G(3, 1) = affine_list[13];
    T_B_G(3, 2) = affine_list[14];
    T_B_G(3, 3) = affine_list[15];

    n_p.param<std::string>("imu_topic", imu_topic, "imu");
    n_p.param<std::string>("joint_state_topic", joint_state_topic, "joint_states");
    n_p.param<std::string>("lfoot_force_torque_topic", lfsr_topic, "force_torque/left");
    n_p.param<std::string>("rfoot_force_torque_topic", rfsr_topic, "force_torque/right");

    n_p.getParam("T_FT_LL", affine_list);
    T_FT_LL(0, 0) = affine_list[0];
    T_FT_LL(0, 1) = affine_list[1];
    T_FT_LL(0, 2) = affine_list[2];
    T_FT_LL(0, 3) = affine_list[3];
    T_FT_LL(1, 0) = affine_list[4];
    T_FT_LL(1, 1) = affine_list[5];
    T_FT_LL(1, 2) = affine_list[6];
    T_FT_LL(1, 3) = affine_list[7];
    T_FT_LL(2, 0) = affine_list[8];
    T_FT_LL(2, 1) = affine_list[9];
    T_FT_LL(2, 2) = affine_list[10];
    T_FT_LL(2, 3) = affine_list[11];
    T_FT_LL(3, 0) = affine_list[12];
    T_FT_LL(3, 1) = affine_list[13];
    T_FT_LL(3, 2) = affine_list[14];
    T_FT_LL(3, 3) = affine_list[15];
    p_FT_LL = Vector3d(T_FT_LL(0, 3), T_FT_LL(1, 3), T_FT_LL(2, 3));

    n_p.getParam("T_FT_RL", affine_list);
    T_FT_RL(0, 0) = affine_list[0];
    T_FT_RL(0, 1) = affine_list[1];
    T_FT_RL(0, 2) = affine_list[2];
    T_FT_RL(0, 3) = affine_list[3];
    T_FT_RL(1, 0) = affine_list[4];
    T_FT_RL(1, 1) = affine_list[5];
    T_FT_RL(1, 2) = affine_list[6];
    T_FT_RL(1, 3) = affine_list[7];
    T_FT_RL(2, 0) = affine_list[8];
    T_FT_RL(2, 1) = affine_list[9];
    T_FT_RL(2, 2) = affine_list[10];
    T_FT_RL(2, 3) = affine_list[11];
    T_FT_RL(3, 0) = affine_list[12];
    T_FT_RL(3, 1) = affine_list[13];
    T_FT_RL(3, 2) = affine_list[14];
    T_FT_RL(3, 3) = affine_list[15];
    p_FT_RL = Vector3d(T_FT_RL(0, 3), T_FT_RL(1, 3), T_FT_RL(2, 3));

  

    //Attitude Estimation for Leg Odometry
    n_p.param<bool>("useMahony", useMahony, true);
    if (useMahony)
    {
        //Mahony Filter for Attitude Estimation
        n_p.param<double>("Mahony_Kp", Kp, 0.25);
        n_p.param<double>("Mahony_Ki", Ki, 0.0);
        mh = new serow::Mahony(freq, Kp, Ki);
    }
    else
    {
        //Madgwick Filter for Attitude Estimation
        n_p.param<double>("Madgwick_gain", beta, 0.012f);
        mw = new serow::Madgwick(freq, beta);
    }
    n_p.param<double>("bias_ax", bias_ax, 0.0);
    n_p.param<double>("bias_ay", bias_ay, 0.0);
    n_p.param<double>("bias_az", bias_az, 0.0);
    n_p.param<double>("bias_gx", bias_gx, 0.0);
    n_p.param<double>("bias_gy", bias_gy, 0.0);
    n_p.param<double>("bias_gz", bias_gz, 0.0);
}






bool humanoid_state_publisher::connect(const ros::NodeHandle nh)
{
    ROS_INFO_STREAM("GEM Data Generator Initializing...");
    // Initialize ROS nodes
    n = nh;
    // Load ROS Parameters
    loadparams();
    //Initialization
    init();
    //Subscribe/Publish ROS Topics/Services
    subscribe();
    advertise();
   
    is_connected_ = true;
    ros::Duration(1.0).sleep();
    ROS_INFO_STREAM("GEM Data Generator Initialized");
    return true;
}

bool humanoid_state_publisher::connected()
{
    return is_connected_;
}

void humanoid_state_publisher::subscribe()
{
    subscribeToIMU();
    subscribeToFSR();
    subscribeToJointState();

}

void humanoid_state_publisher::init()
{
    /** Initialize Variables **/
    //Kinematic TFs
    LLegGRF = Vector3d::Zero();
    RLegGRF = Vector3d::Zero();
    LLegGRT = Vector3d::Zero();
    RLegGRT = Vector3d::Zero();
    abb = Vector3d::Zero();
    wbb = Vector3d::Zero();

    abl = Vector3d::Zero();
    abr = Vector3d::Zero();
    
    omegabl = Vector3d::Zero();
    omegabr = Vector3d::Zero();
    
    vbl = Vector3d::Zero();
    vbr = Vector3d::Zero();
   
    Tbl = Affine3d::Identity();
    Tbr = Affine3d::Identity();


    firstGyrodot = true;
    imu_inc = false;
    lfsr_inc = false;
    rfsr_inc = false;
    joint_inc = false;
}

/** Main Loop **/
void humanoid_state_publisher::run()
{

    static ros::Rate rate(freq); //ROS Node Loop Rate
    while (ros::ok())
    {
        if (imu_inc)
        {
          
            if (useMahony)
            {
                mh->updateIMU(T_B_G.linear() * (Vector3d(imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z) ,
                              );
            }
            else
            {
                mw->updateIMU(T_B_G.linear() * (Vector3d(imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z) + Vector3d(bias_gx, bias_gy, bias_gz)),
                              T_B_A.linear() * (Vector3d(imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y, imu_msg.linear_acceleration.z) + Vector3d(bias_ax, bias_ay, bias_az)));
            }
             //Compute the required transformation matrices (tfs) with Kinematics
            if (joint_inc)
                computeKinTFs();
        }
        ros::spinOnce();
        rate.sleep();
    }
     bodyAcc_est_msg.header.stamp = ros::Time::now();
        bodyAcc_est_msg.header.frame_id = base_link;
        bodyAcc_est_msg.linear_acceleration.x = imuInEKF->accX;
        bodyAcc_est_msg.linear_acceleration.y = imuInEKF->accY;
        bodyAcc_est_msg.linear_acceleration.z = imuInEKF->accZ;

        bodyAcc_est_msg.angular_velocity.x = imuInEKF->gyroX;
        bodyAcc_est_msg.angular_velocity.y = imuInEKF->gyroY;
        bodyAcc_est_msg.angular_velocity.z = imuInEKF->gyroZ;
        bodyAcc_est_pub.publish(bodyAcc_est_msg);
        temp_pose_msg.pose.position.x = Tbl.translation()(0);
        temp_pose_msg.pose.position.y = Tbl.translation()(1);
        temp_pose_msg.pose.position.z = Tbl.translation()(2);
        temp_pose_msg.pose.orientation.x = qbl.x();
        temp_pose_msg.pose.orientation.y = qbl.y();
        temp_pose_msg.pose.orientation.z = qbl.z();
        temp_pose_msg.pose.orientation.w = qbl.w();
        temp_pose_msg.header.stamp = ros::Time::now();
        temp_pose_msg.header.frame_id = base_link_frame;
        rel_leftlegPose_pub.publish(temp_pose_msg);

        temp_pose_msg.pose.position.x = Tbr.translation()(0);
        temp_pose_msg.pose.position.y = Tbr.translation()(1);
        temp_pose_msg.pose.position.z = Tbr.translation()(2);
        temp_pose_msg.pose.orientation.x = qbr.x();
        temp_pose_msg.pose.orientation.y = qbr.y();
        temp_pose_msg.pose.orientation.z = qbr.z();
        temp_pose_msg.pose.orientation.w = qbr.w();

        temp_pose_msg.header.stamp = ros::Time::now();
        temp_pose_msg.header.frame_id = base_link_frame;
        rel_rightlegPose_pub.publish(temp_pose_msg);
}




void humanoid_ekf::computeKinTFs()
{

    //Update the Kinematic Structure
    rd->updateJointConfig(joint_state_pos_map, joint_state_vel_map, joint_noise_density);
    //Get the CoM w.r.t Body Frame
    CoM = rd->comPosition();
    Tbl.translation() = rd->linkPosition(lfoot_frame);
    qbl = rd->linkOrientation(lfoot_frame);
    Tbl.linear() = qbl.toRotationMatrix();

    Tbr.translation() = rd->linkPosition(rfoot_frame);
    qbr = rd->linkOrientation(rfoot_frame);
    Tbr.linear() = qbr.toRotationMatrix();


    //Differential Kinematics with Pinnochio
    omegabl = rd->getAngularVelocity(lfoot_frame);
    omegabr = rd->getAngularVelocity(rfoot_frame);
    vbl = rd->getLinearVelocity(lfoot_frame);
    vbr = rd->getLinearVelocity(rfoot_frame);
    if (!kinematicsInitialized)
        kinematicsInitialized = true;  
}

void humanoid_state_publisher::filterGyrodot()
{
    if (!firstGyrodot)
    {
        //Compute numerical derivative
  
        Gyrodot = (Gyro - Gyro_) * freq;
        
        
        if (useGyroLPF)
        {
            Gyrodot(0) = gyroLPF[0]->filter(Gyrodot(0));
            Gyrodot(1) = gyroLPF[1]->filter(Gyrodot(1));
            Gyrodot(2) = gyroLPF[2]->filter(Gyrodot(2));
        }
        else
        {
            gyroMAF[0]->filter(Gyrodot(0));
            gyroMAF[1]->filter(Gyrodot(1));
            gyroMAF[2]->filter(Gyrodot(2));

            Gyrodot(0) = gyroMAF[0]->x;
            Gyrodot(1) = gyroMAF[1]->x;
            Gyrodot(2) = gyroMAF[2]->x;
        }
    }
    else
    {
        Gyrodot = Vector3d::Zero();
        firstGyrodot = false;
    }
    Gyro_ = Gyro;
}


void humanoid_state_publisher::advertise()
{
    bodyAcc_est_pub = n.advertise<sensor_msgs::Imu>("gem/rel_base_imu", 1000);
    bodyAngularAcceleration_pub = n.advertise<geometry_msgs::TwistStamped>("gem/rel_base_angular_acceleration",1000);
    rel_LFootPose_pub = n.advertise<geometry_msgs::PoseStamped>("gem/rel_LLeg_pose",1000);
    rel_RFootPose_pub = n.advertise<geometry_msgs::PoseStamped>("gem/rel_RLeg_pose",1000);
    rel_LFootTwist_pub = n.advertise<geometry_msgs::TwistStamped>("gem/rel_LLeg_linear_velocity",1000);
    rel_RFootTwist_pub = n.advertise<geometry_msgs::TwistStamped>("gem/rel_RLeg_linear_velocity",1000);
    rel_LFootAcc_pub = n.advertise<geometry_msgs::TwistStamped>("gem/rel_LLeg_linear_acceleration",1000);
    rel_RFootAcc_pub = n.advertise<geometry_msgs::TwistStamped>("gem/rel_RLeg_linear_acceleration",1000);
    joint_states_pub = n.advertise<sensor_msgs::JointState>("gem/joint_states", 1000);
    rel_CoM_pub = n.advertise<geometry_msgs::PointStamped>("gem/rel_CoM",1000);
}

void humanoid_state_publisher::subscribeToJointState()
{

    joint_state_sub = n.subscribe(joint_state_topic, 1, &humanoid_state_publisher::joint_stateCb, this, ros::TransportHints().tcpNoDelay());
    firstJointStates = true;
}

void humanoid_state_publisher::joint_stateCb(const sensor_msgs::JointState::ConstPtr &msg)
{
    joint_state_msg = *msg;
    joint_inc = true;

    if (firstJointStates)
    {
        number_of_joints = joint_state_msg.name.size();
        joint_state_vel.resize(number_of_joints);
        joint_state_pos.resize(number_of_joints);
        JointVF = new JointDF *[number_of_joints];
        for (unsigned int i = 0; i < number_of_joints; i++)
        {
            JointVF[i] = new JointDF();
            JointVF[i]->init(joint_state_msg.name[i], joint_freq, joint_cutoff_freq);
        }
        firstJointStates = false;
    }

    for (unsigned int i = 0; i < joint_state_msg.name.size(); i++)
    {
        joint_state_pos[i] = joint_state_msg.position[i];
        joint_state_vel[i] = JointVF[i]->filter(joint_state_msg.position[i]);
        joint_state_pos_map[joint_state_msg.name[i]] = joint_state_pos[i];
        joint_state_vel_map[joint_state_msg.name[i]] = joint_state_vel[i];
    }

    joint_filt_msg.header.stamp = ros::Time::now();
    joint_filt_msg.name.resize(number_of_joints);
    joint_filt_msg.position.resize(number_of_joints);
    joint_filt_msg.velocity.resize(number_of_joints);

    for (unsigned int i = 0; i < number_of_joints; i++)
    {
        joint_filt_msg.position[i] = JointVF[i]->JointPosition;
        joint_filt_msg.velocity[i] = JointVF[i]->JointVelocity;
        joint_filt_msg.name[i] = JointVF[i]->JointName;
    }

    joint_filt_pub.publish(joint_filt_msg);
}


void humanoid_state_publisher::subscribeToIMU()
{
    imu_sub = n.subscribe(imu_topic, 1, &humanoid_state_publisher::imuCb, this, ros::TransportHints().tcpNoDelay());
}
void humanoid_state_publisher::imuCb(const sensor_msgs::Imu::ConstPtr &msg)
{
    imu_msg = *msg;
    imu_inc = true;
    wbb = T_B_G.linear() * (Vector3d(imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z));
    abb = T_B_A.linear() * (Vector3d(imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y, imu_msg.linear_acceleration.z));


}

void humanoid_state_publisher::subscribeToFSR()
{
    //Left Foot Wrench
    lfsr_sub = n.subscribe(lfsr_topic, 1, &humanoid_state_publisher::lfsrCb, this, ros::TransportHints().tcpNoDelay());
    //Right Foot Wrench
    rfsr_sub = n.subscribe(rfsr_topic, 1, &humanoid_state_publisher::rfsrCb, this, ros::TransportHints().tcpNoDelay());
}

void humanoid_state_publisher::lfsrCb(const geometry_msgs::WrenchStamped::ConstPtr &msg)
{

    if(kinematicsInitialized)
    {
        LLegGRF(0) = msg->wrench.force.x;
        LLegGRF(1) = msg->wrench.force.y;
        LLegGRF(2) = msg->wrench.force.z;
        LLegGRT(0) = msg->wrench.torque.x;
        LLegGRT(1) = msg->wrench.torque.y;
        LLegGRT(2) = msg->wrench.torque.z;
        LLegGRF = Tbl.linear() * T_FT_LL.linear() * LLegGRF;
        LLegGRT = Tbl.linear() * T_FT_LL.linear() * LLegGRT;


        LLeg_est_msg.wrench.force.x  = LLegGRF(0);
        LLeg_est_msg.wrench.force.y  = LLegGRF(1);
        LLeg_est_msg.wrench.force.z  = LLegGRF(2);
        LLeg_est_msg.wrench.torque.x = LLegGRT(0);
        LLeg_est_msg.wrench.torque.y = LLegGRT(1);
        LLeg_est_msg.wrench.torque.z = LLegGRT(2);
        LLeg_est_msg.header.frame_id = base_frame;
        LLeg_est_msg.header.stamp    = ros::Time::now();
        LLeg_est_pub.publish(LLeg_est_msg);

    }  

}

void humanoid_state_publisher::rfsrCb(const geometry_msgs::WrenchStamped::ConstPtr &msg)
{
    rfsr_msg = *msg;


    if(kinematicsInitialized)
    {
        RLegGRF(0) = msg->wrench.force.x;
        RLegGRF(1) = msg->wrench.force.y;
        RLegGRF(2) = msg->wrench.force.z;
        RLegGRT(0) = msg->wrench.torque.x;
        RLegGRT(1) = msg->wrench.torque.y;
        RLegGRT(2) = msg->wrench.torque.z;
        RLegGRF = Tbr.linear() * T_FT_RL.linear() * RLegGRF;
        RLegGRT = Tbr.linear() * T_FT_RL.linear() * RLegGRT;

        RLeg_est_msg.wrench.force.x  = RLegGRF(0);
        RLeg_est_msg.wrench.force.y  = RLegGRF(1);
        RLeg_est_msg.wrench.force.z  = RLegGRF(2);
        RLeg_est_msg.wrench.torque.x = RLegGRT(0);
        RLeg_est_msg.wrench.torque.y = RLegGRT(1);
        RLeg_est_msg.wrench.torque.z = RLegGRT(2);
        RLeg_est_msg.header.frame_id = rfoot_frame;
        RLeg_est_msg.header.stamp    = ros::Time::now();
        RLeg_est_pub.publish(RLeg_est_msg);
    }
    
}