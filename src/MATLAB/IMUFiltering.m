lacc = [laccX,laccY,laccZ];
racc = [raccX,raccY,raccZ];
acc = [accX,accY,accZ];
lgyro = [lgX,lgY,lgZ];
rgyro = [rgX,rgY,rgZ];
gyro = [gX,gY,gZ];

%delay = 9; %Nao
delay = -1; %Talos
fs = 100;
%fc = 25; %Nao
fc = 15; %Talos
[b, a] = butter(2, (2*fc)/fs, 'low');
laccf = filtfilt(b, a, lacc);
raccf = filtfilt(b, a, racc);
figure
plot(laccf(:,3),'red');
hold on
plot(lacc(:,3),'black');
fc = 15;
[b, a] = butter(2, (2*fc)/fs, 'low');
lgyrof = filtfilt(b, a, lgyro);
rgyrof = filtfilt(b, a, rgyro);
figure
plot(rgyrof(:,3),'black');
hold on
plot(rgyro(:,3),'red');
fc = 15;
[b, a] = butter(2, (2*fc)/fs, 'low');
accf = filtfilt(b, a, acc);
figure
plot(acc(:,3),'black');
hold on
plot(accf(:,3),'red');
fc = 15;
[b, a] = butter(2, (2*fc)/fs, 'low');
gyrof = filtfilt(b, a, gyro);
gyrodot = [0 0 0;diff(gyrof)];

if(delay>0)
    accf(:,1)  = delayseq(accf(:,1),delay);
    accf(:,2)  = delayseq(accf(:,2),delay);
    accf(:,3)  = delayseq(accf(:,3),delay);
end

acc_LLegf  = -laccf - cross(gyrodot,lpos) - cross(gyrof,lv);
acc_RLegf  = -raccf - cross(gyrodot,rpos) - cross(gyrof,rv);
g_LLegf = -lgyrof;
g_RLegf = -rgyrof;

acc_LLeg  = -lacc - cross(gyrodot,lpos) - cross(gyro,lv);
acc_RLeg  = -racc - cross(gyrodot,rpos) - cross(gyro,rv);
figure
plot(accf(:,2),'black');
hold on
plot(acc_LLegf(:,2),'red');
hold on
plot(acc_RLegf(:,2),'green');

if(saveData == 1)
    %LLeg Label
    dlmwrite(strcat(saveDir,'/baccX_LL.txt'),acc_LLeg(1:dlen,1))
    dlmwrite(strcat(saveDir,'/baccY_LL.txt'),acc_LLeg(1:dlen,2))
    dlmwrite(strcat(saveDir,'/baccZ_LL.txt'),acc_LLeg(1:dlen,3))
    %RLeg Label
    dlmwrite(strcat(saveDir,'/baccX_RL.txt'),acc_RLeg(1:dlen,1))
    dlmwrite(strcat(saveDir,'/baccY_RL.txt'),acc_RLeg(1:dlen,2))
    dlmwrite(strcat(saveDir,'/baccZ_RL.txt'),acc_RLeg(1:dlen,3))
    %LLeg Label
    dlmwrite(strcat(saveDir,'/baccX_LLf.txt'),acc_LLegf(1:dlen,1))
    dlmwrite(strcat(saveDir,'/baccY_LLf.txt'),acc_LLegf(1:dlen,2))
    dlmwrite(strcat(saveDir,'/baccZ_LLf.txt'),acc_LLegf(1:dlen,3))
    %RLeg Label
    dlmwrite(strcat(saveDir,'/baccX_RLf.txt'),acc_RLegf(1:dlen,1))
    dlmwrite(strcat(saveDir,'/baccY_RLf.txt'),acc_RLegf(1:dlen,2))
    dlmwrite(strcat(saveDir,'/baccZ_RLf.txt'),acc_RLegf(1:dlen,3))
    dlmwrite(strcat(saveDir,'/baccXf.txt'),accf(1:dlen,1))
    dlmwrite(strcat(saveDir,'/baccYf.txt'),accf(1:dlen,2))
    dlmwrite(strcat(saveDir,'/baccZf.txt'),accf(1:dlen,3))
end
