%%
%Set the required paths
saveData = 1;
pathTorosbag = '~/Desktop/icraWS1GEM.bag';
save_dir = 'GEM_test_nao_WS1';
mkdir(save_dir);

%Set the required topics
imu_topic = '/nao_robot/imu';
lft_topic = '/nao_robot/LLeg/force_torque_states';
rft_topic = '/nao_robot/RLeg/force_torque_states';
com_topic = '/SERoW/rel_CoM/pose';
%%
%Import the bagfile
bag=rosbag(pathTorosbag);

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


%CoM 
bagSelection = select(bag,'Topic',com_topic);
test = timeseries(bagSelection,'Pose.Position.X');
c_encx = test.Data;
test = timeseries(bagSelection,'Pose.Position.Y');
c_ency = test.Data;
test = timeseries(bagSelection,'Pose.Position.Z');
c_encz = test.Data;

if(saveData == 1)
    cd(save_dir)
    dlmwrite('gX.txt',gX)
    dlmwrite('gY.txt',gY)
    dlmwrite('gZ.txt',gZ)
    dlmwrite('accX.txt',accX)
    dlmwrite('accY.txt',accY)
    dlmwrite('accZ.txt',accZ) 
    dlmwrite('c_encx.txt',c_encx)
    dlmwrite('c_ency.txt',c_ency)
    dlmwrite('c_encz.txt',c_encz)
    dlmwrite('lfX.txt',lfX)
    dlmwrite('lfY.txt',lfY)
    dlmwrite('lfZ.txt',lfZ)
    dlmwrite('rfX.txt',rfX)
    dlmwrite('rfY.txt',rfY)
    dlmwrite('rfZ.txt',rfZ)
    dlmwrite('ltX.txt',ltX)
    dlmwrite('ltY.txt',ltY)
    dlmwrite('ltZ.txt',ltZ)
    dlmwrite('rtX.txt',rtX)
    dlmwrite('rtY.txt',rtY)
    dlmwrite('rtZ.txt',rtZ)
    cd ..
end


