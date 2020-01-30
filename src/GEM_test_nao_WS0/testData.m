clear all
close all
clc
c_encz = load('accX.txt');
load('lfZ.txt')
load('rfZ.txt')

deltaF = lfZ - rfZ;
%d_encz = [0;diff(c_encz)];
d_encz = c_encz;
%deltaF = [0;diff(deltaF)];
%Standarize
c_std = (c_encz - mean(c_encz))/std(c_encz);
dF_std = (deltaF - mean(deltaF))/std(deltaF);
d_std = (d_encz - mean(d_encz))/std(d_encz);

%Normalize 
c_norm  = (c_encz - min(c_encz)) / (max(c_encz) - min(c_encz));
d_norm  = (d_encz - min(d_encz)) / (max(d_encz) - min(d_encz));
dF_norm  = (deltaF - min(deltaF)) / (max(deltaF) - min(deltaF));

dlen = min(length(deltaF),length(c_encz));
start = 1000;

X=[dF_norm,d_norm];

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
return 
figure
plot(c_std(start:dlen),dF_std(start:dlen))
figure
plot(c_norm(start:dlen),dF_norm(start:dlen))