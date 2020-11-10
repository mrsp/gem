clear all
close all
clc


%%
%Set the required paths
saveData = 1;
useGT = 0;
pathTorosbag = ' ';
saveDir = ' ';

if(saveData == 1)
    if ~exist(saveDir, 'dir')
        mkdir(saveDir);
    else
        delete(strcat(saveDir,'\*'))
    end
end
%Set the required topics
imu_topic = '/gem/rel_base_imu';
lft_topic = '/gem/rel_LLeg_wrench';
rft_topic = '/gem/rel_RLeg_wrench';
vcom_topic = '/gem/rel_CoM_velocity';
ccom_topic = '/gem/rel_CoM_position';
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


%RLeg F/T
bagSelection = select(bag,'Topic',rft_topic);
rft_data = readMessages(bagSelection,1:dlen,'DataFormat','struct');
rf(:,1) = cellfun(@(m) double(m.Wrench.Force.X),rft_data);
rf(:,2) = cellfun(@(m) double(m.Wrench.Force.Y),rft_data);
rf(:,3) = cellfun(@(m) double(m.Wrench.Force.Z),rft_data);
rt(:,1) = cellfun(@(m) double(m.Wrench.Torque.X),rft_data);
rt(:,2) = cellfun(@(m) double(m.Wrench.Torque.Y),rft_data);
rt(:,3) = cellfun(@(m) double(m.Wrench.Torque.Z),rft_data);


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
   lfW(i,:) = (Rotwb{i} * lf(i,:)')';
   ltW(i,:) = (Rotwb{i} * lt(i,:)')';
   vcomW(i,:) = (Rotwb{i} * vcom(i,:)')';
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
