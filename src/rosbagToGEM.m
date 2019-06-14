%%
%Set the required paths and topics
saveData = 0;
pathTorosbag = '~/Desktop/centauro_walk.bag';
imu_topic = '/xbotcore/imu/imu_link';
lft_topic = ' ';
rft_topic = ' ';
com_topic = ' ';
dir = 'GEM_test_cogimon';
mkdir(dir);
%%
%Import the bagfile
bag=rosbag(path);

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
test = timeseries(bagSelection,'Point.X');
c_encx = test.Data;
test = timeseries(bagSelection,'Point.Y');
c_ency = test.Data;
test = timeseries(bagSelection,'Point.Z');
c_encz = test.Data;

if(saveData == 1)
    cd dir
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
end


