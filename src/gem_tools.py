#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
 * GeM - Gait-phase Estimation Module
 *
 * Copyright 2018-2021 Stylianos Piperakis and Stavros Timotheatos, Foundation for Research and Technology Hellas (FORTH)
 * License: BSD
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code self.must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form self.must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Foundation for Research and Technology Hellas (FORTH) 
 *	     nor the names of its contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
'''


import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools
from math import *
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages


#color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
#'darkorange'])


my_colors = [(0.5,0,0.5),(0,0.5,0.5),(0.8,0.36,0.36)]
cmap_name = 'my_list'
my_cmap = LinearSegmentedColormap.from_list(
    cmap_name, my_colors, N=10000)
color_iter = itertools.cycle(my_colors)

params = {
    'axes.labelsize': 15,
    #  'text.fontsize': 15,
    'font.size' : 15,
    'axes.labelsize': 15,
    'axes.titlesize': 15,
    'legend.fontsize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'text.usetex': True,
    'figure.figsize': [7, 4] # instead of 4.5, 4.5
}
plt.rcParams.update(params)

class GeM_tools():
    def __init__(self, validation = False, gt_comparison=False):
        self.gt_comparison = gt_comparison
        self.validation = validation


    def input_data(self, training_path, validation_path):

        rfX = np.loadtxt(training_path+'/rfX.txt')
        rfY = np.loadtxt(training_path+'/rfY.txt')
        rfZ = np.loadtxt(training_path+'/rfZ.txt')
        rtX = np.loadtxt(training_path+'/rtX.txt')
        rtY = np.loadtxt(training_path+'/rtY.txt')
        rtZ = np.loadtxt(training_path+'/rtZ.txt')
        lfX = np.loadtxt(training_path+'/lfX.txt')
        lfY = np.loadtxt(training_path+'/lfY.txt')
        lfZ = np.loadtxt(training_path+'/lfZ.txt')
        ltX = np.loadtxt(training_path+'/ltX.txt')
        ltY = np.loadtxt(training_path+'/ltY.txt')
        ltZ = np.loadtxt(training_path+'/ltZ.txt')
        dlen = min(np.size(lfZ),np.size(rfZ))
        gX = np.loadtxt(training_path+'/gX.txt')
        gY = np.loadtxt(training_path+'/gY.txt')
        gZ = np.loadtxt(training_path+'/gZ.txt')
        accX = np.loadtxt(training_path+'/accX.txt')
        accY = np.loadtxt(training_path+'/accY.txt')
        accZ = np.loadtxt(training_path+'/accZ.txt')
        dlen = min(dlen,np.size(accZ))
        dcX = np.loadtxt(training_path+'/comvX.txt')
        dcY = np.loadtxt(training_path+'/comvY.txt')
        dcZ = np.loadtxt(training_path+'/comvZ.txt')
        dlen = min(dlen,np.size(dcZ))      

        if(self.gt_comparison):
            #if(self.gem2):
            phase = np.loadtxt(training_path+'/gt.txt')
            dlen = min(dlen,np.size(phase))

        if(self.validation):
            rfX_val = np.loadtxt(validation_path+'/rfX.txt')
            rfY_val = np.loadtxt(validation_path+'/rfY.txt')
            rfZ_val = np.loadtxt(validation_path+'/rfZ.txt')
            rtX_val = np.loadtxt(validation_path+'/rtX.txt')
            rtY_val = np.loadtxt(validation_path+'/rtY.txt')
            rtZ_val = np.loadtxt(validation_path+'/rtZ.txt')
            lfX_val = np.loadtxt(validation_path+'/lfX.txt')
            lfY_val = np.loadtxt(validation_path+'/lfY.txt')
            lfZ_val = np.loadtxt(validation_path+'/lfZ.txt')
            ltX_val = np.loadtxt(validation_path+'/ltX.txt')
            ltY_val = np.loadtxt(validation_path+'/ltY.txt')
            ltZ_val = np.loadtxt(validation_path+'/ltZ.txt')
            dlen_val = min(np.size(lfZ_val),np.size(rfZ_val))
            gX_val = np.loadtxt(validation_path+'/gX.txt')
            gY_val = np.loadtxt(validation_path+'/gY.txt')
            gZ_val = np.loadtxt(validation_path+'/gZ.txt')
            accX_val = np.loadtxt(validation_path+'/accX.txt')
            accY_val = np.loadtxt(validation_path+'/accY.txt')
            accZ_val = np.loadtxt(validation_path+'/accZ.txt')
            dlen_val = min(dlen_val,np.size(accZ_val))
            dcX_val = np.loadtxt(validation_path+'/comvX.txt')
            dcY_val = np.loadtxt(validation_path+'/comvY.txt')
            dcZ_val = np.loadtxt(validation_path+'/comvZ.txt')
            dlen_val = min(dlen_val,np.size(dcZ_val))            
            


        self.data_train = np.array([])
        self.data_val = np.array([])

        #Leg Forces and Torques
        self.data_train = lfX[0:dlen] - rfX[0:dlen]
        self.data_train = np.column_stack([self.data_train, lfY[0:dlen] - rfY[0:dlen]])
        self.data_train = np.column_stack([self.data_train, lfZ[0:dlen] - rfZ[0:dlen]])
        self.data_train = np.column_stack([self.data_train, ltX[0:dlen] - rtX[0:dlen]])
        self.data_train = np.column_stack([self.data_train, ltY[0:dlen] - rtY[0:dlen]])
        self.data_train = np.column_stack([self.data_train, ltZ[0:dlen] - rtZ[0:dlen]])

        #CoM Velocity
        self.data_train = np.column_stack([self.data_train, dcX[0:dlen]])
        self.data_train = np.column_stack([self.data_train, dcY[0:dlen]])
        self.data_train = np.column_stack([self.data_train, dcZ[0:dlen]])
       
        #Base Linear Acceleration and Base Angular Velocity
        self.data_train = np.column_stack([self.data_train, accX[0:dlen]])
        self.data_train = np.column_stack([self.data_train, accY[0:dlen]])
        self.data_train = np.column_stack([self.data_train, accZ[0:dlen]])
        self.data_train = np.column_stack([self.data_train, gX[0:dlen]])
        self.data_train = np.column_stack([self.data_train, gY[0:dlen]])
        self.data_train = np.column_stack([self.data_train, gZ[0:dlen]])


        self.data_train_min = np.zeros((self.data_train.shape[1]))
        self.data_train_max = np.zeros((self.data_train.shape[1]))
        self.data_train_mean = np.zeros((self.data_train.shape[1]))
        self.data_train_std = np.zeros((self.data_train.shape[1]))
    
        #Data Statistics
        for i in range(self.data_train.shape[1]):
            self.data_train_min[i] = np.min(self.data_train[:, i])
            self.data_train_max[i] = np.max(self.data_train[:, i])
            self.data_train_mean[i] = np.mean(self.data_train[:, i])
            self.data_train_std[i] = np.std(self.data_train[:, i])
            self.data_train[:, i] = self.normalize_data(self.data_train[:, i],self.data_train_max[i], self.data_train_min[i])   
            #self.data_train[:, i] = self.standarize_data(self.data_train[:, i],self.data_train_mean[i], self.data_train_std[i])   
            #self.data_train[:, i] = self.normalizeMean_data(self.data_train[:, i],self.data_train_max[i], self.data_train_min[i],self.data_train_mean[i])   

        '''
        plt.plot(self.data_label[:,1], color = [0.5,0.5,0.5])
        plt.plot(self.data_label[:,4], color = [0,0.5,0.5])
        plt.plot(self.data_label[:,7], color = [0.8,0.36,0.36])
        plt.grid('on')
        plt.show()
       '''

        if(self.validation):
            #Leg Forces and Torques
            self.data_val = lfX_val[0:dlen_val] - rfX_val[0:dlen_val]
            self.data_val = np.column_stack([self.data_val, lfY_val[0:dlen_val] - rfY_val[0:dlen_val]])
            self.data_val = np.column_stack([self.data_val, lfZ_val[0:dlen_val] - rfZ_val[0:dlen_val]])
            self.data_val = np.column_stack([self.data_val, ltX_val[0:dlen_val] - rtX_val[0:dlen_val]])
            self.data_val = np.column_stack([self.data_val, ltY_val[0:dlen_val] - rtY_val[0:dlen_val]])
            self.data_val = np.column_stack([self.data_val, ltZ_val[0:dlen_val] - rtZ_val[0:dlen_val]])

            #CoM Velocity
            self.data_val = np.column_stack([self.data_val, dcX_val[0:dlen_val]])
            self.data_val = np.column_stack([self.data_val, dcY_val[0:dlen_val]])
            self.data_val = np.column_stack([self.data_val, dcZ_val[0:dlen_val]])

            #Base Linear Acceleration and Base Angular Velocity
            self.data_val = np.column_stack([self.data_val, accX_val[0:dlen_val]])
            self.data_val = np.column_stack([self.data_val, accY_val[0:dlen_val]])
            self.data_val = np.column_stack([self.data_val, accZ_val[0:dlen_val]])
            self.data_val = np.column_stack([self.data_val, gX_val[0:dlen_val]])
            self.data_val = np.column_stack([self.data_val, gY_val[0:dlen_val]])
            self.data_val = np.column_stack([self.data_val, gZ_val[0:dlen_val]])



            self.data_val_min = np.zeros((self.data_val.shape[1]))
            self.data_val_max = np.zeros((self.data_val.shape[1]))
            self.data_val_mean = np.zeros((self.data_val.shape[1]))
            self.data_val_std = np.zeros((self.data_val.shape[1]))
        
            #Data Statistics
            for i in range(self.data_val.shape[1]):
                self.data_val_min[i] = np.min(self.data_val[:, i])
                self.data_val_max[i] = np.max(self.data_val[:, i])
                self.data_val_mean[i] = np.mean(self.data_val[:, i])
                self.data_val_std[i] = np.std(self.data_val[:, i])
                self.data_val[:, i] = self.normalize_data(self.data_val[:, i],self.data_val_max[i], self.data_val_min[i])   
                #self.data_val[:, i] = self.standarize_data(self.data_val[:, i],self.data_val_mean[i], self.data_val_std[i])   
                #self.data_val[:, i] = self.normalizeMean_data(self.data_val[:, i],self.data_val_max[i], self.data_val_min[i],self.data_val_mean[i])   

        
        if (self.gt_comparison):
            self.phase = phase[0:dlen]
            self.dlen = dlen
            '''
            else:
                phase2=np.append([phase],[np.zeros_like(np.arange(cX.shape[0]-phase.shape[0]))])
                self.cX = cX[~(phase2==-1)]
                self.cY = cY[~(phase2==-1)]
                self.cZ = cZ[~(phase2==-1)]
                phase3=np.append([phase],[np.zeros_like(np.arange(accX.shape[0]-phase.shape[0]))])
                self.accX = accX[~(phase3==-1)]
                self.accY = accY[~(phase3==-1)]
                self.accZ = accZ[~(phase3==-1)]
                phase4=np.append([phase],[np.zeros_like(np.arange(gX.shape[0]-phase.shape[0]))])
                self.gX = gX[~(phase4==-1)]
                self.gY = gY[~(phase4==-1)]
                phase5=np.append([phase],[np.zeros_like(np.arange(lfZ.shape[0]-phase.shape[0]))])
                self.lfZ = lfZ[~(phase5==-1)]
                self.lfX = lfX[~(phase5==-1)]
                self.lfY = lfY[~(phase5==-1)]
                phase6=np.append([phase],[np.zeros_like(np.arange(rfZ.shape[0]-phase.shape[0]))])
                self.rfZ = rfZ[~(phase6==-1)]
                self.rfX = rfX[~(phase6==-1)]
                self.rfY = rfY[~(phase6==-1)]
                phase7=np.append([phase],[np.zeros_like(np.arange(ltZ.shape[0]-phase.shape[0]))])
                self.ltZ = ltZ[~(phase7==-1)]
                self.ltX = ltX[~(phase7==-1)]
                self.ltY = ltY[~(phase7==-1)]
                phase8=np.append([phase],[np.zeros_like(np.arange(rtZ.shape[0]-phase.shape[0]))])
                self.rtZ = rtZ[~(phase8==-1)]
                self.rtX = rtX[~(phase8==-1)]
                self.rtY = rtY[~(phase8==-1)]
                self.data_train=self.data_train[~(phase==-1)]
                self.phase=phase[~(phase==-1)]
                self.dlen = np.size(self.phase)
            '''
        else:
            self.dlen = dlen
        

      

        print("Data Dim")
        print(self.dlen)


    def genInput(self, data, gt=None):

        if gt is None:
            gt=self

        output_ = np.array([])
        output_ = np.append(output_, data.lfX - data.rfX, axis = 0)
        output_ = np.append(output_, data.lfY - data.rfY, axis = 0)
        output_ = np.append(output_, data.lfZ - data.rfZ, axis = 0)
        output_ = np.append(output_, data.ltX - data.rtX, axis = 0)
        output_ = np.append(output_, data.ltY - data.rtY, axis = 0)
        output_ = np.append(output_, data.ltZ - data.rtZ, axis = 0)
        output_ = np.append(output_, data.dcX, axis = 0)
        output_ = np.append(output_, data.dcY, axis = 0)
        output_ = np.append(output_, data.dcZ, axis = 0)
        output_ = np.append(output_, data.accX, axis = 0)
        output_ = np.append(output_, data.accY, axis = 0)
        output_ = np.append(output_, data.accZ, axis = 0)
        output_ = np.append(output_, data.gX, axis = 0)
        output_ = np.append(output_, data.gY, axis = 0)
        output_ = np.append(output_, data.gZ, axis = 0)

        for i in range(self.data_train.shape[1]):
            output_[i] = self.normalize_data(output_[i],self.data_train_max[i], self.data_train_min[i])   

        return output_


    def normalize_data(self,din, dmax, dmin, min_range=-1, max_range = 1):    
        if(dmax-dmin != 0):
            dout = min_range  + (max_range - min_range) * (din - dmin)/(dmax - dmin)
        else:
            dout =  np.zeros((np.size(din)))

        return dout

    def standarize_data(self,din,dmean,dstd):
        if(dstd != 0):
            dout = (din - dmean)/dstd
        else:
            dout =  np.zeros((np.size(din)))

        return dout


    def normalize(self,din, dmax, dmin, min_range=-1, max_range = 1):    
        if(din>dmax):
            din=dmax
        elif(din<dmin):
            din=dmin

        if(dmax-dmin != 0):
            dout = min_range  + (max_range - min_range) * (din - dmin)/(dmax - dmin)
        else:
            dout =  0

        return dout

    def normalizeMean(self,din, dmax, dmin, dmean):    
        if(din>dmax):
            din=dmax
        elif(din<dmin):
            din=dmin

        if(dmax-dmin != 0):
            dout = (din - dmean)/(dmax-dmin)
        else:
            dout =  0

        return dout

    def normalizeMean_data(self,din, dmax, dmin, dmean):    
        if(dmax-dmin != 0):
            dout = (din - dmean)/(dmax-dmin)
        else:
            dout =  np.zeros((np.size(din)))

        return dout  

    def standarize(self,din,dmean,dstd):
        if(dstd != 0):
            dout = (din - dmean)/dstd
        else:
            dout =  0

        return dout


    def genGroundTruthStatistics(self, reduced_data):
        if(self.gt_comparison):
            #remove extra zeros elements
            d1 = np.zeros((self.dlen,2))
            d2 = np.zeros((self.dlen,2))
            d3 = np.zeros((self.dlen,2))
            
            for i in range(self.dlen):
                if (self.phase[i]==0):
                    d1[i,0] = reduced_data[i,0]
                    d1[i,1] = reduced_data[i,1]
                elif (self.phase[i]==1):
                    d2[i,0] = reduced_data[i,0]
                    d2[i,1] = reduced_data[i,1]
                elif (self.phase[i]==2):
                    d3[i,0] = reduced_data[i,0]
                    d3[i,1] = reduced_data[i,1]

            d1=d1[~(d1==0).all(1)]
            d2=d2[~(d2==0).all(1)]
            d3=d3[~(d3==0).all(1)]
            print('----')
            print(d1)
            print('----')
            print('----')
            print(d2)
            print('----')
            print('----')
            print(d3)
            print('----')
            mean=np.zeros((3,2))
            mean[0,0]=np.mean(d1[:,0])
            mean[0,1]=np.mean(d1[:,1])
            mean[1,0]=np.mean(d2[:,0])
            mean[1,1]=np.mean(d2[:,1])
            mean[2,0]=np.mean(d3[:,0])
            mean[2,1]=np.mean(d3[:,1])

            print(mean)

            self.mean = mean
            covariance1=np.cov(d1.T)
            covariance2=np.cov(d2.T)
            covariance3=np.cov(d3.T)
            self.covariance=(covariance1, covariance2, covariance3)
        else:
            print('Input data did not have Ground-Truth Information')




    def plot_results(self,X, Y_, means, covariances, title):
        fig = plt.figure()
        splot = plt.subplot(1, 1, 1)


        if(covariances is not None):
            for i, (mean, covar, color) in enumerate(zip(
                    means, covariances, color_iter)):
                v, w = linalg.eigh(covar)
                v = 2. * np.sqrt(2.) * np.sqrt(v)
                u = w[0] / linalg.norm(w[0])
                # as the DP will not use every component it has access to
                # unless it needs it, we shouldn't plot the redundant
                # components.
                # Plot an ellipse to show the Gaussian component
                angle = np.arctan(u[1] / u[0])
                angle = 180. * angle / np.pi  # convert to degrees
                ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color='w',linestyle='dashed',linewidth='2.0',ec='w')
                ell.set_clip_box(splot.bbox)
                ell.set_alpha(0.5)
                ell.set_fill(False)
                splot.add_artist(ell)

                if not np.any(Y_ == i):
                    continue
                plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, cmap=my_cmap)
        else:
            for i, (mean, color) in enumerate(zip(
                    means, color_iter)):
                if not np.any(Y_ == i):
                    continue
                plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, cmap=my_cmap)


        plt.scatter(means[:, 0], means[:, 1],
                    marker='x', s=169, linewidths=3,
                    color='red', zorder=10)
        plt.title(title)
        plt.grid('on')
        plt.show()



    def plot_latent_space(self,g):
        plt.scatter(g.reduced_data_train[:,0],g.reduced_data_train[:,1],.8)
        if(g.pca_dim):
            plt.title(" ")
        else:
            plt.title(" ")
        plt.grid('on')
        plt.show()


    def plot_confusion_matrix(self,cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):


        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, [classes[2],classes[1],classes[0]], rotation=45)
        plt.yticks(tick_marks, [classes[2],classes[1],classes[0]], rotation=45)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        cm_linesum=np.sum(cm.round(2),axis=1)
        diff_cm=1-cm_linesum
        add_cm=np.zeros_like(cm)+np.diag(diff_cm)
        cm=cm+add_cm
        #        print cm_linesum
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

class GEM_data:
    def __init__(self):
        self.lfX = 0
        self.lfY = 0
        self.lfZ = 0
        self.ltX = 0
        self.ltY = 0
        self.ltZ = 0
        self.rfX = 0
        self.rfY = 0
        self.rfZ = 0
        self.rtX = 0
        self.rtY = 0
        self.rtZ = 0
        self.accX = 0
        self.accY = 0
        self.accZ = 0
        self.gX = 0
        self.gY = 0
        self.gZ = 0
        self.dcX = 0
        self.dcY = 0
        self.dcZ = 0