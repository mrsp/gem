clear all
close all
clc
c_encz = load('c_ency.txt');
load('lfZ.txt')
load('rfZ.txt')

deltaF = lfZ - rfZ;
deltaF = c_encz;
fc = 1.0; % Cut off frequency
fs = 100; % Sampling rate
start = 1;

[b,a] = butter(2,fc/(fs/2),'high'); % Butterworth filter of order 6
deltaFF = filter(b,a,deltaF); % Will be the filtered signal


%d_encz = [0;diff(c_encz)];
d_encz = deltaFF;
%deltaF = [0;diff(deltaF)];
%Standarize
c_std = (c_encz(start:end) - mean(c_encz(start:end)))/std(c_encz(start:end));
dF_std = (deltaF(start:end) - mean(deltaF(start:end)))/std(deltaF(start:end));
d_std = (d_encz(start:end) - mean(d_encz(start:end)))/std(d_encz(start:end));

%Normalize 
c_norm  = (c_encz(start:end) - min(c_encz(start:end))) / (max(c_encz(start:end)) - min(c_encz(start:end)));
d_norm  = (d_encz(start:end) - min(d_encz(start:end))) / (max(d_encz(start:end)) - min(d_encz(start:end)));
dF_norm  = (deltaF(start:end) - min(deltaF(start:end))) / (max(deltaF(start:end)) - min(deltaF(start:end)));

dlen = min(length(deltaF),length(c_encz));

X=[dF_norm,d_norm];





plot(deltaF(1:2500),'black')
hold on
plot(deltaFF(1:2500),'--')















return 




[idx,C] = kmeans(X,3);
x1 = min(X(:,1)):0.01:max(X(:,1));
x2 = min(X(:,2)):0.01:max(X(:,2));
[x1G,x2G] = meshgrid(x1,x2);
XGrid = [x1G(:),x2G(:)]; % Defines a fine grid on the plot

idx2Region = kmeans(XGrid,3,'MaxIter',1000,'Start',C);
figure;
gscatter(XGrid(:,1),XGrid(:,2),idx2Region,...
    [0,0.75,0.75;0.75,0,0.75;0.75,0.75,0],'..');
hold on;
plot(X(:,1),X(:,2),'k*','MarkerSize',5);
title 'GEM Data';
xlabel 'X_2';
ylabel 'X_1'; 
legend('LSS','DS','RSS','Data','Location','SouthEast');
hold off;


figure;
plot(X(idx==1,1),X(idx==1,2),'r.','MarkerSize',12)
hold on
plot(X(idx==2,1),X(idx==2,2),'b.','MarkerSize',12)
plot(C(:,1),C(:,2),'kx',...
     'MarkerSize',15,'LineWidth',3) 
legend('Cluster 1','Cluster 2','Centroids',...
       'Location','NW')
title 'Cluster Assignments and Centroids'
hold off
figure
plot(c_std(start:dlen),dF_std(start:dlen))
figure
plot(c_norm(start:dlen),dF_norm(start:dlen))