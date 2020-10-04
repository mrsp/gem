data_dir = 'TALOS_GEM/';

%Leg Forces
lfX = load(strcat(data_dir, 'lfX.txt'));
lfY = load(strcat(data_dir, 'lfY.txt'));
lfZ = load(strcat(data_dir, 'lfZ.txt'));
rfX = load(strcat(data_dir, 'rfX.txt'));
rfY = load(strcat(data_dir, 'rfY.txt'));
rfZ = load(strcat(data_dir, 'rfZ.txt'));

lfX =  normalize_data(lfX,-1,1);
lfY =  normalize_data(lfY,-1,1);
lfX =  normalize_data(lfX,-1,1);
rfX =  normalize_data(rfX,-1,1);
rfY =  normalize_data(rfY,-1,1);
rfX =  normalize_data(rfX,-1,1);


%Leg Torques
ltX = load(strcat(data_dir, 'ltX.txt'));
ltY = load(strcat(data_dir, 'ltY.txt'));
ltZ = load(strcat(data_dir, 'ltZ.txt'));
rtX = load(strcat(data_dir, 'rtX.txt'));
rtY = load(strcat(data_dir, 'rtY.txt'));
rtZ = load(strcat(data_dir, 'rtZ.txt'));

ltX =  normalize_data(ltX,-1,1);
ltY =  normalize_data(ltY,-1,1);
ltX =  normalize_data(ltX,-1,1);
rtX =  normalize_data(rtX,-1,1);
rtY =  normalize_data(rtY,-1,1);
rtX =  normalize_data(rtX,-1,1);

%Leg Velocities
lvX = load(strcat(data_dir, 'lvX.txt'));
lvY = load(strcat(data_dir, 'lvY.txt'));
lvZ = load(strcat(data_dir, 'lvZ.txt'));
rvX = load(strcat(data_dir, 'rvX.txt'));
rvY = load(strcat(data_dir, 'rvY.txt'));
rvZ = load(strcat(data_dir, 'rvZ.txt'));

lvX =  normalize_data(lvX,-1,1);
lvY =  normalize_data(lvY,-1,1);
lvX =  normalize_data(lvX,-1,1);
rvX =  normalize_data(rvX,-1,1);
rvY =  normalize_data(rvY,-1,1);
rvX =  normalize_data(rvX,-1,1);

lwX = load(strcat(data_dir, 'lwX.txt'));
lwY = load(strcat(data_dir, 'lwY.txt'));
lwZ = load(strcat(data_dir, 'lwZ.txt'));
rwX = load(strcat(data_dir, 'rwX.txt'));
rwY = load(strcat(data_dir, 'rwY.txt'));
rwZ = load(strcat(data_dir, 'rwZ.txt'));

lwX =  normalize_data(lwX,-1,1);
lwY =  normalize_data(lwY,-1,1);
lwX =  normalize_data(lwX,-1,1);
rwX =  normalize_data(rwX,-1,1);
rwY =  normalize_data(rwY,-1,1);
rwX =  normalize_data(rwX,-1,1);

%Leg IMUs
laccX = load(strcat(data_dir, 'laccX.txt'));
laccY = load(strcat(data_dir, 'laccY.txt'));
laccZ = load(strcat(data_dir, 'laccZ.txt'));
lgX = load(strcat(data_dir, 'lgX.txt'));
lgY = load(strcat(data_dir, 'lgY.txt'));
lgZ = load(strcat(data_dir, 'lgZ.txt'));


laccX =  normalize_data(laccX,-1,1);
laccY =  normalize_data(laccY,-1,1);
laccZ =  normalize_data(laccZ,-1,1);
lgX =  normalize_data(lgX,-1,1);
lgY =  normalize_data(lgY,-1,1);
lgZ =  normalize_data(lgZ,-1,1);

raccX = load(strcat(data_dir, 'raccX.txt'));
raccY = load(strcat(data_dir, 'raccY.txt'));
raccZ = load(strcat(data_dir, 'raccZ.txt'));
rgX = load(strcat(data_dir, 'rgX.txt'));
rgY = load(strcat(data_dir, 'rgY.txt'));
rgZ = load(strcat(data_dir, 'rgZ.txt'));


raccX =  normalize_data(raccX,-1,1);
raccY =  normalize_data(raccY,-1,1);
raccZ =  normalize_data(raccZ,-1,1);
rgX =  normalize_data(rgX,-1,1);
rgY =  normalize_data(rgY,-1,1);
rgZ =  normalize_data(rgZ,-1,1);

%Base IMU
accX = load(strcat(data_dir, 'accX.txt'));
accY = load(strcat(data_dir, 'accY.txt'));
accZ = load(strcat(data_dir, 'accZ.txt'));
gX = load(strcat(data_dir, 'gX.txt'));
gY = load(strcat(data_dir, 'gY.txt'));
gZ = load(strcat(data_dir, 'gZ.txt'));

accX =  normalize_data(accX,-1,1);
accY =  normalize_data(accY,-1,1);
accZ =  normalize_data(accZ,-1,1);
gX =  normalize_data(gX,-1,1);
gY =  normalize_data(gY,-1,1);
gZ =  normalize_data(gZ,-1,1);


%CoM Velocity
comvX =  load(strcat(data_dir, 'comvX.txt'));
comvY =  load(strcat(data_dir, 'comvY.txt'));
comvZ =  load(strcat(data_dir, 'comvZ.txt'));
comvX =  normalize_data(comvX,-1,1);
comvY =  normalize_data(comvY,-1,1);
comvZ =  normalize_data(comvZ,-1,1);
%GT 
gt = load(strcat(data_dir, 'gt.txt'));

%Label Data
baccX = load(strcat(data_dir, 'baccX.txt'));
baccY = load(strcat(data_dir, 'baccY.txt'));
baccZ = load(strcat(data_dir, 'baccZ.txt'));
baccX =  normalize_data_mean(baccX);
baccY =  normalize_data_mean(baccY);
baccZ =  normalize_data_mean(baccZ);
baccX_LL = load(strcat(data_dir, 'baccX_LL.txt'));
baccY_LL = load(strcat(data_dir, 'baccY_LL.txt'));
baccZ_LL = load(strcat(data_dir, 'baccZ_LL.txt'));

baccX_LL =  normalize_data_mean(baccX_LL);
baccY_LL =  normalize_data_mean(baccY_LL);
baccZ_LL =  normalize_data_mean(baccZ_LL);

baccX_RL = load(strcat(data_dir, 'baccX_RL.txt'));
baccY_RL = load(strcat(data_dir, 'baccY_RL.txt'));
baccZ_RL = load(strcat(data_dir, 'baccZ_RL.txt'));
baccX_RL =  normalize_data_mean(baccX_RL);
baccY_RL =  normalize_data_mean(baccY_RL);
baccZ_RL =  normalize_data_mean(baccZ_RL);



    dvX = lvX - rvX;
    dvY = lvY - rvY;
    dvZ = lvZ - rvZ;
    dwX = lwX - rwX;
    dwY = lwY - rwY;
    dwZ = lwZ - rwZ;
    dfX = lfX - rfX;
    dfY = lfY - rfY;
    dfZ = lfZ - rfZ;
    dtX = ltX - rtX;
    dtY = ltY - rtY;
    dtZ = ltZ - rtZ;

