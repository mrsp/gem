clear all
close all
clc
%%
%Set the required paths
saveData = 1;
useGT = 0;
pathTorosbag = 'C:\Users\stpip\Desktop\nao_first.bag';
save_dir = '../NAO_GEM2';
mkdir(save_dir);

%Set the required topics
imu_topic = '/gem/rel_base_imu2';
orientation_topic = '/gem/rel_base_imu2';
lft_topic = '/gem/rel_LLeg_wrench';
rft_topic = '/gem/rel_RLeg_wrench';
limu_topic = '/gem/rel_LLeg_imu';
rimu_topic = '/gem/rel_RLeg_imu';
%com_topic = '/gem/rel_CoM_position';
vcom_topic = '/gem/rel_CoM_velocity';
lvel_topic = '/gem/rel_LLeg_velocity';
rvel_topic = '/gem/rel_RLeg_velocity';
llabel_topic = '/gem/rel_base_acceleration_LLeg';
rlabel_topic = '/gem/rel_base_acceleration_RLeg';
gt_topic = '/gem/ground_truth/gait_phase';


%%
%Import the bagfile
bag=rosbag(pathTorosbag);
datalen = min(bag.AvailableTopics.NumMessages);

%GT Gait-Phase
if(useGT  == 1)
    bagSelection = select(bag,'Topic',gt_topic);
    test = timeseries(bagSelection,'Data');
    gt = test.Data;
end
%Base IMU
bagSelection = select(bag,'Topic',imu_topic);
test = timeseries(bagSelection,'AngularVelocity.X');
gX = test.Data;
test = timeseries(bagSelection,'AngularVelocity.Y');
gY = test.Data;
test = timeseries(bagSelection,'AngularVelocity.Z');
gZ = test.Data;
test = timeseries(bagSelection,'LinearAcceleration.X');
accX = test.Data;
test = timeseries(bagSelection,'LinearAcceleration.Y');
accY = test.Data;
test = timeseries(bagSelection,'LinearAcceleration.Z');
accZ = test.Data;

bagSelection = select(bag,'Topic',orientation_topic);
test = timeseries(bagSelection,'Orientation.X');
wX = test.Data;
test = timeseries(bagSelection,'Orientation.Y');
wY = test.Data;
test = timeseries(bagSelection,'Orientation.Z');
wZ = test.Data;
test = timeseries(bagSelection,'Orientation.W');
wW = test.Data;

v0 = [gX,gY,gZ];
v1 = [accX,accY,accZ];
for i=1:datalen
   Rotwb{i}=quat2rotm([wW(i),wX(i),wY(i),wZ(i)]);
   %Save inside the loop to have same lengths as datalen
   bgX(i,1) = gX(i,1);
   bgY(i,1) = gY(i,1);
   bgZ(i,1) = gZ(i,1);
   baccX(i,1) = accX(i,1);
   baccY(i,1) = accY(i,1);
   baccZ(i,1) = accZ(i,1);
   %Transform to world frame
   v0(i,:) = (Rotwb{i} * v0(i,:)')';
   v1(i,:) = (Rotwb{i} * v1(i,:)')';
end
gX = v0(:,1);
gY = v0(:,2);
gZ = v0(:,3);
accX = v1(:,1);
accY = v1(:,2);
accZ = v1(:,3);

%Left Foot IMU
bagSelection = select(bag,'Topic',limu_topic);
test = timeseries(bagSelection,'AngularVelocity.X');
lgX = test.Data;
test = timeseries(bagSelection,'AngularVelocity.Y');
lgY = test.Data;
test = timeseries(bagSelection,'AngularVelocity.Z');
lgZ = test.Data;
test = timeseries(bagSelection,'LinearAcceleration.X');
laccX = test.Data;
test = timeseries(bagSelection,'LinearAcceleration.Y');
laccY = test.Data;
test = timeseries(bagSelection,'LinearAcceleration.Z');
laccZ = test.Data;

v0 = [lgX,lgY,lgZ];
v1 = [laccX,laccY,laccZ];
for i=1:datalen
   v0(i,:) = (Rotwb{i} * v0(i,:)')';
   v1(i,:) = (Rotwb{i} * v1(i,:)')';
end
lgX = v0(:,1);
lgY = v0(:,2);
lgZ = v0(:,3);
laccX = v1(:,1);
laccY = v1(:,2);
laccZ = v1(:,3);


%Right Foot IMU
bagSelection = select(bag,'Topic',rimu_topic);
test = timeseries(bagSelection,'AngularVelocity.X');
rgX = test.Data;
test = timeseries(bagSelection,'AngularVelocity.Y');
rgY = test.Data;
test = timeseries(bagSelection,'AngularVelocity.Z');
rgZ = test.Data;
test = timeseries(bagSelection,'LinearAcceleration.X');
raccX = test.Data;
test = timeseries(bagSelection,'LinearAcceleration.Y');
raccY = test.Data;
test = timeseries(bagSelection,'LinearAcceleration.Z');
raccZ = test.Data;

v0 = [rgX,rgY,rgZ];
v1 = [raccX,raccY,raccZ];
for i=1:datalen
   v0(i,:) = (Rotwb{i} * v0(i,:)')';
   v1(i,:) = (Rotwb{i} * v1(i,:)')';
end
rgX = v0(:,1);
rgY = v0(:,2);
rgZ = v0(:,3);
raccX = v1(:,1);
raccY = v1(:,2);
raccZ = v1(:,3);


%Left Leg Forces
bagSelection = select(bag,'Topic',lft_topic);
test = timeseries(bagSelection,'Wrench.Force.X');
lfX = test.Data;
test = timeseries(bagSelection,'Wrench.Force.Y');
lfY = test.Data;
test = timeseries(bagSelection,'Wrench.Force.Z');
lfZ = test.Data;

%Left Leg Torques
test = timeseries(bagSelection,'Wrench.Torque.X');
ltX = test.Data;
test = timeseries(bagSelection,'Wrench.Torque.Y');
ltY = test.Data;
test = timeseries(bagSelection,'Wrench.Torque.Z');
ltZ = test.Data;

v0 = [lfX,lfY,lfZ];
v1 = [ltX,ltY,ltZ];
for i=1:datalen
   v0(i,:) = (Rotwb{i} * v0(i,:)')';
   v1(i,:) = (Rotwb{i} * v1(i,:)')';
end
lfX = v0(:,1);
lfY = v0(:,2);
lfZ = v0(:,3);
ltX = v1(:,1);
ltY = v1(:,2);
ltZ = v1(:,3);


%Right Leg Forces
bagSelection = select(bag,'Topic',rft_topic);
test = timeseries(bagSelection,'Wrench.Force.X');
rfX = test.Data;
test = timeseries(bagSelection,'Wrench.Force.Y');
rfY = test.Data;
test = timeseries(bagSelection,'Wrench.Force.Z');
rfZ = test.Data;

%Right Leg Torques
test = timeseries(bagSelection,'Wrench.Torque.X');
rtX = test.Data;
test = timeseries(bagSelection,'Wrench.Torque.Y');
rtY = test.Data;
test = timeseries(bagSelection,'Wrench.Torque.Z');
rtZ = test.Data;


v0 = [rfX,rfY,rfZ];
v1 = [rtX,rtY,rtZ];
for i=1:datalen
   v0(i,:) = (Rotwb{i} * v0(i,:)')';
   v1(i,:) = (Rotwb{i} * v1(i,:)')';
end
rfX = v0(:,1);
rfY = v0(:,2);
rfZ = v0(:,3);
rtX = v1(:,1);
rtY = v1(:,2);
rtZ = v1(:,3);

%CoM Velocity
bagSelection = select(bag,'Topic',vcom_topic);
test = timeseries(bagSelection,'Twist.Linear.X');
vcomX = test.Data;
test = timeseries(bagSelection,'Twist.Linear.Y');
vcomY = test.Data;
test = timeseries(bagSelection,'Twist.Linear.Z');
vcomZ = test.Data;

v1 = [vcomX,vcomY,vcomZ];
for i=1:datalen
   v1(i,:) = (Rotwb{i} * v1(i,:)')';
end
vcomX = v1(:,1);
vcomY = v1(:,2);
vcomZ = v1(:,3);

%LLeg Velocity
bagSelection = select(bag,'Topic',lvel_topic);
test = timeseries(bagSelection,'Twist.Linear.X');
lvX = test.Data;
test = timeseries(bagSelection,'Twist.Linear.Y');
lvY = test.Data;
test = timeseries(bagSelection,'Twist.Linear.Z');
lvZ = test.Data;

test = timeseries(bagSelection,'Twist.Angular.X');
lwX = test.Data;
test = timeseries(bagSelection,'Twist.Angular.Y');
lwY = test.Data;
test = timeseries(bagSelection,'Twist.Angular.Z');
lwZ = test.Data;

v0 = [lvX,lvY,lvZ];
v1 = [lwX,lwY,lwZ];
for i=1:datalen
   v0(i,:) = (Rotwb{i} * v0(i,:)')';
   v1(i,:) = (Rotwb{i} * v1(i,:)')';
end
lvX = v0(:,1);
lvY = v0(:,2);
lvZ = v0(:,3);
lwX = v1(:,1);
lwY = v1(:,2);
lwZ = v1(:,3);



%RLeg Velocity
bagSelection = select(bag,'Topic',rvel_topic);
test = timeseries(bagSelection,'Twist.Linear.X');
rvX = test.Data;
test = timeseries(bagSelection,'Twist.Linear.Y');
rvY = test.Data;
test = timeseries(bagSelection,'Twist.Linear.Z');
rvZ = test.Data;

test = timeseries(bagSelection,'Twist.Angular.X');
rwX = test.Data;
test = timeseries(bagSelection,'Twist.Angular.Y');
rwY = test.Data;
test = timeseries(bagSelection,'Twist.Angular.Z');
rwZ = test.Data;

v0 = [rvX,rvY,rvZ];
v1 = [rwX,rwY,rwZ];
for i=1:datalen
   v0(i,:) = (Rotwb{i} * v0(i,:)')';
   v1(i,:) = (Rotwb{i} * v1(i,:)')';
end
rvX = v0(:,1);
rvY = v0(:,2);
rvZ = v0(:,3);
rwX = v1(:,1);
rwY = v1(:,2);
rwZ = v1(:,3);

%LLeg contribution to Local Base Acceleration
bagSelection = select(bag,'Topic',llabel_topic);
test = timeseries(bagSelection,'Twist.Linear.X');
baccX_LL = test.Data;
test = timeseries(bagSelection,'Twist.Linear.Y');
baccY_LL = test.Data;
test = timeseries(bagSelection,'Twist.Linear.Z');
baccZ_LL = test.Data;


%LLeg contribution to Local Base Acceleration
bagSelection = select(bag,'Topic',rlabel_topic);
test = timeseries(bagSelection,'Twist.Linear.X');
baccX_RL = test.Data;
test = timeseries(bagSelection,'Twist.Linear.Y');
baccY_RL = test.Data;
test = timeseries(bagSelection,'Twist.Linear.Z');
baccZ_RL = test.Data;



if(saveData == 1)
    cd(save_dir)
    if(useGT == 1)
        dlmwrite('gt.txt',gt)
    end
    
    %Base IMU
    dlmwrite('gX.txt',gX)
    dlmwrite('gY.txt',gY)
    dlmwrite('gZ.txt',gZ)
    dlmwrite('accX.txt',accX)
    dlmwrite('accY.txt',accY)
    dlmwrite('accZ.txt',accZ) 
    %Right Leg IMU
    dlmwrite('rgX.txt',rgX)
    dlmwrite('rgY.txt',rgY)
    dlmwrite('rgZ.txt',rgZ)
    dlmwrite('raccX.txt',raccX)
    dlmwrite('raccY.txt',raccY)
    dlmwrite('raccZ.txt',raccZ) 
    %Right Leg Velocity
    dlmwrite('rvX.txt',rvX)
    dlmwrite('rvY.txt',rvY)
    dlmwrite('rvZ.txt',rvZ)
    dlmwrite('rwX.txt',rwX)
    dlmwrite('rwY.txt',rwY)
    dlmwrite('rwZ.txt',rwZ) 
    %Left Leg IMU
    dlmwrite('lgX.txt',lgX)
    dlmwrite('lgY.txt',lgY)
    dlmwrite('lgZ.txt',lgZ)
    dlmwrite('laccX.txt',laccX)
    dlmwrite('laccY.txt',laccY)
    dlmwrite('laccZ.txt',laccZ)    
    %Left Leg Velocity
    dlmwrite('lvX.txt',lvX)
    dlmwrite('lvY.txt',lvY)
    dlmwrite('lvZ.txt',lvZ)
    dlmwrite('lwX.txt',lwX)
    dlmwrite('lwY.txt',lwY)
    dlmwrite('lwZ.txt',lwZ) 
    %Left Leg F/T
    dlmwrite('lfX.txt',lfX)
    dlmwrite('lfY.txt',lfY)
    dlmwrite('lfZ.txt',lfZ)
    dlmwrite('ltX.txt',ltX)
    dlmwrite('ltY.txt',ltY)
    dlmwrite('ltZ.txt',ltZ)
    %Right Leg F/T
    dlmwrite('rfX.txt',rfX)
    dlmwrite('rfY.txt',rfY)
    dlmwrite('rfZ.txt',rfZ)
    dlmwrite('rtX.txt',rtX)
    dlmwrite('rtY.txt',rtY)
    dlmwrite('rtZ.txt',rtZ)
    %CoM Velocity
    dlmwrite('comvX.txt',vcomX)
    dlmwrite('comvY.txt',vcomY)
    dlmwrite('comvZ.txt',vcomZ)
    %LLeg Label
    dlmwrite('baccX_LL.txt',baccX_LL)
    dlmwrite('baccY_LL.txt',baccY_LL)
    dlmwrite('baccZ_LL.txt',baccZ_LL)
    %RLeg Label
    dlmwrite('baccX_RL.txt',baccX_RL)
    dlmwrite('baccY_RL.txt',baccY_RL)
    dlmwrite('baccZ_RL.txt',baccZ_RL)
    %Base IMU local frame
    dlmwrite('bgX.txt',bgX)
    dlmwrite('bgY.txt',bgY)
    dlmwrite('bgZ.txt',bgZ)
    dlmwrite('baccX.txt',baccX)
    dlmwrite('baccY.txt',baccY)
    dlmwrite('baccZ.txt',baccZ) 
    cd ..
end


