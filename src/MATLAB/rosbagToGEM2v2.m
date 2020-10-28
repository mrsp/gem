clear all
close all
clc


%%
%Set the required paths
saveData = 1;
useGT = 1;
pathTorosbag = 'C:\Users\stpip\Desktop\NEW_TALOS_DATA\talos_27_10_1.bag';
%saveDir = '../NAO_GEM2/IMU_INTEL/SimpleMotions/back2';
saveDir = 'C:\Users\stpip\Desktop\NEW_TALOS_DATA\TALOS3';

if ~exist(saveDir, 'dir')
    mkdir(saveDir);
else
    delete(strcat(saveDir,'\*'))
end
%Set the required topics
imu_topic = '/gem/rel_base_imu';
lft_topic = '/gem/rel_LLeg_wrench';
rft_topic = '/gem/rel_RLeg_wrench';
limu_topic = '/gem/rel_LLeg_imu';
rimu_topic = '/gem/rel_RLeg_imu';
com_topic = '/gem/rel_CoM_position';
vcom_topic = '/gem/rel_CoM_velocity';
lvel_topic = '/gem/rel_LLeg_velocity';
rvel_topic = '/gem/rel_RLeg_velocity';
lpose_topic = '/gem/rel_LLeg_pose';
rpose_topic = '/gem/rel_RLeg_pose';
gt_topic = '/gem/ground_truth/gait_phase';



%Import the bagfile
bag=rosbag(pathTorosbag);
dlen = min(bag.AvailableTopics.NumMessages(5:end))
%GT Gait-Phase
if(useGT  == 1)
    bagSelection = select(bag,'Topic',gt_topic);
    test = timeseries(bagSelection,'Data');
    gt = test.Data;
end
%Body IMU
bagSelection = select(bag,'Topic',imu_topic);
imu_data = readMessages(bagSelection,1:dlen,'DataFormat','struct');
%Body Angular Rate
gyro(:,1) = cellfun(@(m) double(m.AngularVelocity.X),imu_data);
gyro(:,2) = cellfun(@(m) double(m.AngularVelocity.Y),imu_data);
gyro(:,3) = cellfun(@(m) double(m.AngularVelocity.Z),imu_data);
%Body Acceleration
acc(:,1) = cellfun(@(m) double(m.LinearAcceleration.X),imu_data);
acc(:,2) = cellfun(@(m) double(m.LinearAcceleration.Y),imu_data);
acc(:,3) = cellfun(@(m) double(m.LinearAcceleration.Z),imu_data);
%Body Orientation
q(:,1) = cellfun(@(m) double(m.Orientation.W),imu_data);
q(:,2) = cellfun(@(m) double(m.Orientation.X),imu_data);
q(:,3) = cellfun(@(m) double(m.Orientation.Y),imu_data);
q(:,4) = cellfun(@(m) double(m.Orientation.Z),imu_data);

%LLeg F/T
bagSelection = select(bag,'Topic',lft_topic);
lft_data = readMessages(bagSelection,1:dlen,'DataFormat','struct');
%LLeg GRF
lf(:,1) = cellfun(@(m) double(m.Wrench.Force.X),lft_data);
lf(:,2) = cellfun(@(m) double(m.Wrench.Force.Y),lft_data);
lf(:,3) = cellfun(@(m) double(m.Wrench.Force.Z),lft_data);
%LLeg GRT
lt(:,1) = cellfun(@(m) double(m.Wrench.Torque.X),lft_data);
lt(:,2) = cellfun(@(m) double(m.Wrench.Torque.Y),lft_data);
lt(:,3) = cellfun(@(m) double(m.Wrench.Torque.Z),lft_data);
%LLeg Relative Pose
bagSelection = select(bag,'Topic',lpose_topic);
lpose_data = readMessages(bagSelection,1:dlen,'DataFormat','struct');
%LLeg Relative Translation
lpos(:,1) = cellfun(@(m) double(m.Pose.Position.X),lpose_data);
lpos(:,2) = cellfun(@(m) double(m.Pose.Position.Y),lpose_data);
lpos(:,3) = cellfun(@(m) double(m.Pose.Position.Z),lpose_data);
%LLeg Relative Orientation
lorient(:,1) = cellfun(@(m) double(m.Pose.Orientation.W),lpose_data);
lorient(:,2) = cellfun(@(m) double(m.Pose.Orientation.X),lpose_data);
lorient(:,3) = cellfun(@(m) double(m.Pose.Orientation.Y),lpose_data);
lorient(:,4) = cellfun(@(m) double(m.Pose.Orientation.Z),lpose_data); 
%LLeg Vel
bagSelection = select(bag,'Topic',lvel_topic);
lvel_data = readMessages(bagSelection,1:dlen,'DataFormat','struct');
%LLeg Relative Linear Velocity
lv(:,1) = cellfun(@(m) double(m.Twist.Linear.X),lvel_data);
lv(:,2) = cellfun(@(m) double(m.Twist.Linear.Y),lvel_data);
lv(:,3) = cellfun(@(m) double(m.Twist.Linear.Z),lvel_data);
%LLeg Relative Angular Velocity
lw(:,1) =  cellfun(@(m) double(m.Twist.Angular.X),lvel_data);
lw(:,2) =  cellfun(@(m) double(m.Twist.Angular.Y),lvel_data);
lw(:,3) =  cellfun(@(m) double(m.Twist.Angular.Z),lvel_data);

%LLeg IMU
bagSelection = select(bag,'Topic',limu_topic);
limu_data = readMessages(bagSelection,1:dlen,'DataFormat','struct');
%LLeg Relative Linear Acceleration
lacc(:,1) = cellfun(@(m) double(m.LinearAcceleration.X),limu_data);
lacc(:,2) = cellfun(@(m) double(m.LinearAcceleration.Y),limu_data);
lacc(:,3) = cellfun(@(m) double(m.LinearAcceleration.Z),limu_data);
%LLeg Relative Angular Velocity
lgyro(:,1) = cellfun(@(m) double(m.AngularVelocity.X),limu_data);
lgyro(:,2) = cellfun(@(m) double(m.AngularVelocity.Y),limu_data);
lgyro(:,3) = cellfun(@(m) double(m.AngularVelocity.Z),limu_data);
%LLeg Orientation
lq(:,1) = cellfun(@(m) double(m.Orientation.W),limu_data);
lq(:,2) = cellfun(@(m) double(m.Orientation.X),limu_data);
lq(:,3) = cellfun(@(m) double(m.Orientation.Y),limu_data);
lq(:,4) = cellfun(@(m) double(m.Orientation.Z),limu_data);


%RLeg F/T
bagSelection = select(bag,'Topic',rft_topic);
rft_data = readMessages(bagSelection,1:dlen,'DataFormat','struct');
rf(:,1) = cellfun(@(m) double(m.Wrench.Force.X),rft_data);
rf(:,2) = cellfun(@(m) double(m.Wrench.Force.Y),rft_data);
rf(:,3) = cellfun(@(m) double(m.Wrench.Force.Z),rft_data);
rt(:,1) = cellfun(@(m) double(m.Wrench.Torque.X),rft_data);
rt(:,2) = cellfun(@(m) double(m.Wrench.Torque.Y),rft_data);
rt(:,3) = cellfun(@(m) double(m.Wrench.Torque.Z),rft_data);
%RLeg Pose
bagSelection = select(bag,'Topic',rpose_topic);
rpose_data = readMessages(bagSelection,1:dlen,'DataFormat','struct');
rpos(:,1) = cellfun(@(m) double(m.Pose.Position.X),rpose_data);
rpos(:,2) = cellfun(@(m) double(m.Pose.Position.Y),rpose_data);
rpos(:,3) = cellfun(@(m) double(m.Pose.Position.Z),rpose_data);
%RLeg Relative Orientation
rorient(:,1) = cellfun(@(m) double(m.Pose.Orientation.W),rpose_data);
rorient(:,2) = cellfun(@(m) double(m.Pose.Orientation.X),rpose_data);
rorient(:,3) = cellfun(@(m) double(m.Pose.Orientation.Y),rpose_data);
rorient(:,4) = cellfun(@(m) double(m.Pose.Orientation.Z),rpose_data); 
%RLeg Vel
bagSelection = select(bag,'Topic',rvel_topic);
rvel_data = readMessages(bagSelection,1:dlen,'DataFormat','struct');
rv(:,1) = cellfun(@(m) double(m.Twist.Linear.X),rvel_data);
rv(:,2) = cellfun(@(m) double(m.Twist.Linear.Y),rvel_data);
rv(:,3) = cellfun(@(m) double(m.Twist.Linear.Z),rvel_data);
rw(:,1) =  cellfun(@(m) double(m.Twist.Angular.X),rvel_data);
rw(:,2) =  cellfun(@(m) double(m.Twist.Angular.Y),rvel_data);
rw(:,3) =  cellfun(@(m) double(m.Twist.Angular.Z),rvel_data);

%RLeg IMU
bagSelection = select(bag,'Topic',rimu_topic);
rimu_data = readMessages(bagSelection,1:dlen,'DataFormat','struct');
racc(:,1) = cellfun(@(m) double(m.LinearAcceleration.X),rimu_data);
racc(:,2) = cellfun(@(m) double(m.LinearAcceleration.Y),rimu_data);
racc(:,3) = cellfun(@(m) double(m.LinearAcceleration.Z),rimu_data);
rgyro(:,1) = cellfun(@(m) double(m.AngularVelocity.X),rimu_data);
rgyro(:,2) = cellfun(@(m) double(m.AngularVelocity.Y),rimu_data);
rgyro(:,3) = cellfun(@(m) double(m.AngularVelocity.Z),rimu_data);
%RLeg Orientation
rq(:,1) = cellfun(@(m) double(m.Orientation.W),rimu_data);
rq(:,2) = cellfun(@(m) double(m.Orientation.X),rimu_data);
rq(:,3) = cellfun(@(m) double(m.Orientation.Y),rimu_data);
rq(:,4) = cellfun(@(m) double(m.Orientation.Z),rimu_data);

bagSelection = select(bag,'Topic',com_topic);
com_data = readMessages(bagSelection,1:dlen,'DataFormat','struct');
com(:,1) = cellfun(@(m) double(m.Point.X),com_data);
com(:,2) = cellfun(@(m) double(m.Point.Y),com_data);
com(:,3) = cellfun(@(m) double(m.Point.Z),com_data);

bagSelection = select(bag,'Topic',vcom_topic);
vcom_data = readMessages(bagSelection,1:dlen,'DataFormat','struct');
vcom(:,1) = cellfun(@(m) double(m.Twist.Linear.X),vcom_data);
vcom(:,2) = cellfun(@(m) double(m.Twist.Linear.Y),vcom_data);
vcom(:,3) = cellfun(@(m) double(m.Twist.Linear.Z),vcom_data);


%Transform Vectors to World Frame
for i=1:dlen
   Rotwb{i}=quat2rotm(q(i,:));
   accW(i,:) = (Rotwb{i} * acc(i,:)')';
   gyroW(i,:) = (Rotwb{i} * gyro(i,:)')';

   rfW(i,:) = (Rotwb{i} * rf(i,:)')';
   rtW(i,:) = (Rotwb{i} * rt(i,:)')';
   rposW(i,:) = (Rotwb{i} * rpos(i,:)')';
   rvW(i,:) = (Rotwb{i} * rv(i,:)')';
   rwW(i,:) = (Rotwb{i} * rw(i,:)')';
   
   Rotwr{i}=quat2rotm(rq(i,:));

   raccW(i,:) = (Rotwr{i} * racc(i,:)')';
   rgyroW(i,:) = (Rotwr{i} * rgyro(i,:)')';
   
   lfW(i,:) = (Rotwb{i} * lf(i,:)')';
   ltW(i,:) = (Rotwb{i} * lt(i,:)')';
   lposW(i,:) = (Rotwb{i} * lpos(i,:)')';
   lvW(i,:) = (Rotwb{i} * lv(i,:)')';
   lwW(i,:) = (Rotwb{i} * lw(i,:)')';

   vcomW(i,:) = (Rotwb{i} * vcom(i,:)')';

   Rotwl{i}=quat2rotm(lq(i,:));
   laccW(i,:) = (Rotwl{i} * lacc(i,:)')';
   lgyroW(i,:) = (Rotwl{i} * lgyro(i,:)')';
   
end

if(saveData == 1)
    if(useGT == 1)
        dlmwrite(strcat(saveDir,'/gt.txt'),gt)
    end
    %Base IMU
    dlmwrite(strcat(saveDir,'/gX.txt'),gyroW(1:dlen,1))
    dlmwrite(strcat(saveDir,'/gY.txt'),gyroW(1:dlen,2))
    dlmwrite(strcat(saveDir,'/gZ.txt'),gyroW(1:dlen,3))
    dlmwrite(strcat(saveDir,'/accX.txt'),accW(1:dlen,1))
    dlmwrite(strcat(saveDir,'/accY.txt'),accW(1:dlen,2))
    dlmwrite(strcat(saveDir,'/accZ.txt'),accW(1:dlen,3)) 
    %Right Leg IMU
    dlmwrite(strcat(saveDir,'/rgX.txt'),rgyroW(1:dlen,1))
    dlmwrite(strcat(saveDir,'/rgY.txt'),rgyroW(1:dlen,2))
    dlmwrite(strcat(saveDir,'/rgZ.txt'),rgyroW(1:dlen,3))
    dlmwrite(strcat(saveDir,'/raccX.txt'),raccW(1:dlen,1))
    dlmwrite(strcat(saveDir,'/raccY.txt'),raccW(1:dlen,2))
    dlmwrite(strcat(saveDir,'/raccZ.txt'),raccW(1:dlen,3)) 
    %Right Leg Velocity
    dlmwrite(strcat(saveDir,'/rvX.txt'),rvW(1:dlen,1))
    dlmwrite(strcat(saveDir,'/rvY.txt'),rvW(1:dlen,2))
    dlmwrite(strcat(saveDir,'/rvZ.txt'),rvW(1:dlen,3))
    dlmwrite(strcat(saveDir,'/rwX.txt'),rwW(1:dlen,1))
    dlmwrite(strcat(saveDir,'/rwY.txt'),rwW(1:dlen,2))
    dlmwrite(strcat(saveDir,'/rwZ.txt'),rwW(1:dlen,3)) 
    %Left Leg IMU
    dlmwrite(strcat(saveDir,'/lgX.txt'),lgyroW(1:dlen,1))
    dlmwrite(strcat(saveDir,'/lgY.txt'),lgyroW(1:dlen,2))
    dlmwrite(strcat(saveDir,'/lgZ.txt'),lgyroW(1:dlen,3))
    dlmwrite(strcat(saveDir,'/laccX.txt'),laccW(1:dlen,1))
    dlmwrite(strcat(saveDir,'/laccY.txt'),laccW(1:dlen,2))
    dlmwrite(strcat(saveDir,'/laccZ.txt'),laccW(1:dlen,3))    
    %Left Leg Velocity
    dlmwrite(strcat(saveDir,'/lvX.txt'),lvW(1:dlen,1))
    dlmwrite(strcat(saveDir,'/lvY.txt'),lvW(1:dlen,2))
    dlmwrite(strcat(saveDir,'/lvZ.txt'),lvW(1:dlen,3))
    dlmwrite(strcat(saveDir,'/lwX.txt'),lwW(1:dlen,1))
    dlmwrite(strcat(saveDir,'/lwY.txt'),lwW(1:dlen,2))
    dlmwrite(strcat(saveDir,'/lwZ.txt'),lwW(1:dlen,3)) 
    %Left Leg F/T
    dlmwrite(strcat(saveDir,'/lfX.txt'),lfW(1:dlen,1))
    dlmwrite(strcat(saveDir,'/lfY.txt'),lfW(1:dlen,2))
    dlmwrite(strcat(saveDir,'/lfZ.txt'),lfW(1:dlen,3))
    dlmwrite(strcat(saveDir,'/ltX.txt'),ltW(1:dlen,1))
    dlmwrite(strcat(saveDir,'/ltY.txt'),ltW(1:dlen,2))
    dlmwrite(strcat(saveDir,'/ltZ.txt'),ltW(1:dlen,3))
    %Right Leg F/T
    dlmwrite(strcat(saveDir,'/rfX.txt'),rfW(1:dlen,1))
    dlmwrite(strcat(saveDir,'/rfY.txt'),rfW(1:dlen,2))
    dlmwrite(strcat(saveDir,'/rfZ.txt'),rfW(1:dlen,3))
    dlmwrite(strcat(saveDir,'/rtX.txt'),rtW(1:dlen,1))
    dlmwrite(strcat(saveDir,'/rtY.txt'),rtW(1:dlen,2))
    dlmwrite(strcat(saveDir,'/rtZ.txt'),rtW(1:dlen,3))
    %CoM Velocity
    dlmwrite(strcat(saveDir,'/comvX.txt'),vcomW(1:dlen,1))
    dlmwrite(strcat(saveDir,'/comvY.txt'),vcomW(1:dlen,2))
    dlmwrite(strcat(saveDir,'/comvZ.txt'),vcomW(1:dlen,3))
end
talosFiltering
%NAOFiltering