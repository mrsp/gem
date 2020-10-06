clear all
close all
loadData

OPTIONS(1) = 1.2;
OPTIONS(2) = 10000;
OPTIONS(3) = 1e-7;
OPTIONS(4) = 1.0;

LLeg_dataX = [lfX,lfY,lfZ,ltX,ltY,ltZ,laccX];
LLeg_dataY = [lfX,lfY,lfZ,ltX,ltY,ltZ,laccY];
LLeg_dataZ = [lfX,lfY,lfZ,ltX,ltY,ltZ,laccZ];
LLeg_dataRoll = [lfX,lfY,lfZ,ltX,ltY,ltZ,lgX];
LLeg_dataPitch = [lfX,lfY,lfZ,ltX,ltY,ltZ,lgY];
LLeg_dataYaw = [lfX,lfY,lfZ,ltX,ltY,ltZ,lgZ];




[LLeg_centerX,LLeg_probX,~] = fcm(LLeg_dataX,2,OPTIONS);
[LLeg_centerY,LLeg_probY,~] = fcm(LLeg_dataY,2,OPTIONS);
[LLeg_centerZ,LLeg_probZ,~] = fcm(LLeg_dataZ,2,OPTIONS);
[LLeg_centerRoll,LLeg_probRoll,~] = fcm(LLeg_dataRoll,2,OPTIONS);
[LLeg_centerPitch,LLeg_probPitch,~] = fcm(LLeg_dataPitch,2,OPTIONS);
[LLeg_centerYaw,LLeg_probYaw,~] = fcm(LLeg_dataYaw,2,OPTIONS);

LLeg_prob(1,:) = max(LLeg_probX) .* max(LLeg_probY) .* max(LLeg_probZ) .* max(LLeg_probRoll) .* max(LLeg_probPitch) .* max(LLeg_probYaw);
LLeg_prob(2,:) = min(LLeg_probX) .* min(LLeg_probY) .* min(LLeg_probZ) .* min(LLeg_probRoll) .* min(LLeg_probPitch) .* min(LLeg_probYaw);





RLeg_dataX = [rfX,rfY,rfZ,rtX,rtY,rtZ,raccX];
RLeg_dataY = [rfX,rfY,rfZ,rtX,rtY,rtZ,raccY];
RLeg_dataZ = [rfX,rfY,rfZ,rtX,rtY,rtZ,raccZ];
RLeg_dataRoll = [rfX,rfY,rfZ,rtX,rtY,rtZ,rgX];
RLeg_dataPitch = [rfX,rfY,rfZ,rtX,rtY,rtZ,rgY];
RLeg_dataYaw = [rfX,rfY,rfZ,rtX,rtY,rtZ,rgZ];




[RLeg_centerX,RLeg_probX,~] = fcm(RLeg_dataX,2,OPTIONS);
[RLeg_centerY,RLeg_probY,~] = fcm(RLeg_dataY,2,OPTIONS);
[RLeg_centerZ,RLeg_probZ,~] = fcm(RLeg_dataZ,2,OPTIONS);
[RLeg_centerRoll,RLeg_probRoll,~] = fcm(RLeg_dataRoll,2,OPTIONS);
[RLeg_centerPitch,RLeg_probPitch,~] = fcm(RLeg_dataPitch,2,OPTIONS);
[RLeg_centerYaw,RLeg_probYaw,~] = fcm(RLeg_dataYaw,2,OPTIONS);

RLeg_prob(1,:) = max(RLeg_probX) .* max(RLeg_probY) .* max(RLeg_probZ) .* max(RLeg_probRoll) .* max(RLeg_probPitch) .* max(RLeg_probYaw);
RLeg_prob(2,:) = min(RLeg_probX) .* min(RLeg_probY) .* min(RLeg_probZ) .* min(RLeg_probRoll) .* min(RLeg_probPitch) .* min(RLeg_probYaw);

accX_est = LLeg_prob(1,:)'.*baccX_LL + RLeg_prob(1,:)'.*baccX_RL;
accY_est = LLeg_prob(1,:)'.*baccY_LL + RLeg_prob(1,:)'.*baccY_RL;
accZ_est = LLeg_prob(1,:)'.*baccZ_LL + RLeg_prob(1,:)'.*baccZ_RL;


maxU = max([LLeg_prob(1,:); RLeg_prob(1,:)]);
indexL = find(LLeg_prob(1,:) == maxU);
indexR = find(RLeg_prob(1,:) == maxU);
figure
%plot(indexL, accX_est(indexL), 'g--', 'LineWidth', 0.1);
%hold on;
%plot(indexR, accX_est(indexR), 'r--', 'LineWidth', 0.1);
%hold on
plot(accX_est, 'r--', 'LineWidth', 0.1);
hold
plot(baccX,'black')
error = 0.4 * abs(baccX - accX_est) + 0.4 * abs(baccY - accY_est) +  0.2 * abs(baccZ - accZ_est);
error = mean(error)

