#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
 * GeM - Gait-phase Estimation Module
 *
 * Copyright 2018-2020 Stylianos Piperakis and Stavros Timotheatos, Foundation for Research and Technology Hellas (FORTH)
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
    def __init__(self, validation = False, gt_comparison=False, gem2 = False, useLabels = False):
        self.gt_comparison = gt_comparison
        self.validation = validation
        self.gem2 = gem2
        self.useLabels = useLabels


    def input_data(self, training_path, validation_path):
        
        if(self.gt_comparison):
            #if(self.gem2):
            phase = np.loadtxt(training_path+'/gt.txt')
            dlen = np.size(phase)
            '''
            else:
                gt_lfZ  = np.loadtxt(training_path+'/gt_lfZ.txt')
                gt_rfZ  = np.loadtxt(training_path+'/gt_rfZ.txt')
                gt_lfX  = np.loadtxt(training_path+'/gt_lfX.txt')
                gt_rfX  = np.loadtxt(training_path+'/gt_rfX.txt')
                gt_lfY  = np.loadtxt(training_path+'/gt_lfY.txt')
                gt_rfY  = np.loadtxt(training_path+'/gt_rfY.txt')
                mu  = np.loadtxt(training_path+'/mu.txt')
                dlen = min(dlen,min(np.size(gt_rfZ),np.size(gt_lfZ)))
                self.mu = mu
                lcon = np.sqrt(gt_lfX[i] * gt_lfX[i] + gt_lfY[i] * gt_lfY[i])
                rcon = np.sqrt(gt_rfX[i] * gt_rfX[i] + gt_rfY[i] * gt_rfY[i])
                if( ((self.mu[i]*gt_lfZ[i])>lcon) and ((self.mu[i] * gt_rfZ[i])>rcon)):
                    phase[i] = 0
                elif( (self.mu[i]*gt_lfZ[i])>lcon ):
                    phase[i] = 1
                elif( (self.mu[i]*gt_rfZ[i])>rcon ):
                    phase[i] = 2
                else:
                    phase[i] = -1
            '''



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

        if(self.gem2):
            lvX = np.loadtxt(training_path+'/lvX.txt')
            lvY = np.loadtxt(training_path+'/lvY.txt')
            lvZ = np.loadtxt(training_path+'/lvZ.txt')
            dlen = min(dlen,np.size(lvZ))
            rvX = np.loadtxt(training_path+'/rvX.txt')
            rvY = np.loadtxt(training_path+'/rvY.txt')
            rvZ = np.loadtxt(training_path+'/rvZ.txt')
            dlen = min(dlen,np.size(rvZ))
            lwX = np.loadtxt(training_path+'/lwX.txt')
            lwY = np.loadtxt(training_path+'/lwY.txt')
            lwZ = np.loadtxt(training_path+'/lwZ.txt')
            rwX = np.loadtxt(training_path+'/rwX.txt')
            rwY = np.loadtxt(training_path+'/rwY.txt')
            rwZ = np.loadtxt(training_path+'/rwZ.txt')
            laccX = np.loadtxt(training_path+'/laccX.txt')
            laccY = np.loadtxt(training_path+'/laccY.txt')
            laccZ = np.loadtxt(training_path+'/laccZ.txt')
            dlen = min(dlen,np.size(laccZ))
            raccX = np.loadtxt(training_path+'/raccX.txt')
            raccY = np.loadtxt(training_path+'/raccY.txt')
            raccZ = np.loadtxt(training_path+'/raccZ.txt')
            dlen = min(dlen,np.size(raccZ))
   
        if(self.useLabels):
            baccX_LL = np.loadtxt(training_path+'/baccX_LL.txt')
            baccY_LL = np.loadtxt(training_path+'/baccY_LL.txt')
            baccZ_LL = np.loadtxt(training_path+'/baccZ_LL.txt')
            baccX_RL = np.loadtxt(training_path+'/baccX_RL.txt')
            baccY_RL = np.loadtxt(training_path+'/baccY_RL.txt')
            baccZ_RL = np.loadtxt(training_path+'/baccZ_RL.txt')
            baccX = np.loadtxt(training_path+'/baccX.txt')
            baccY = np.loadtxt(training_path+'/baccY.txt')
            baccZ = np.loadtxt(training_path+'/baccZ.txt')

            bgX_LL = np.loadtxt(training_path+'/bgX_LL.txt')
            bgY_LL = np.loadtxt(training_path+'/bgY_LL.txt')
            bgZ_LL = np.loadtxt(training_path+'/bgZ_LL.txt')
            bgX_RL = np.loadtxt(training_path+'/bgX_RL.txt')
            bgY_RL = np.loadtxt(training_path+'/bgY_RL.txt')
            bgZ_RL = np.loadtxt(training_path+'/bgZ_RL.txt')
            bgX = np.loadtxt(training_path+'/bgX.txt')
            bgY = np.loadtxt(training_path+'/bgY.txt')
            bgZ = np.loadtxt(training_path+'/bgZ.txt')
            dlen = min(dlen,min(np.size(bgZ_LL),np.size(baccZ_RL)))
       


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
            dlen = min(np.size(lfZ_val),np.size(rfZ_val))
            gX_val = np.loadtxt(validation_path+'/gX.txt')
            gY_val = np.loadtxt(validation_path+'/gY.txt')
            gZ_val = np.loadtxt(validation_path+'/gZ.txt')
            accX_val = np.loadtxt(validation_path+'/accX.txt')
            accY_val = np.loadtxt(validation_path+'/accY.txt')
            accZ_val = np.loadtxt(validation_path+'/accZ.txt')
            dlen = min(dlen,np.size(accZ_val))
            dcX_val = np.loadtxt(validation_path+'/comvX.txt')
            dcY_val = np.loadtxt(validation_path+'/comvY.txt')
            dcZ_val = np.loadtxt(validation_path+'/comvZ.txt')
            dlen = min(dlen,np.size(dcZ_val))            

            if(self.gem2):
                lvX_val = np.loadtxt(validation_path+'/lvX.txt')
                lvY_val = np.loadtxt(validation_path+'/lvY.txt')
                lvZ_val = np.loadtxt(validation_path+'/lvZ.txt')
                dlen = min(dlen,np.size(lvZ_val))
                rvX_val = np.loadtxt(validation_path+'/rvX.txt')
                rvY_val = np.loadtxt(validation_path+'/rvY.txt')
                rvZ_val = np.loadtxt(validation_path+'/rvZ.txt')
                dlen = min(dlen,np.size(rvZ_val))
                lwX_val = np.loadtxt(validation_path+'/lwX.txt')
                lwY_val = np.loadtxt(validation_path+'/lwY.txt')
                lwZ_val = np.loadtxt(validation_path+'/lwZ.txt')
                rwX_val = np.loadtxt(validation_path+'/rwX.txt')
                rwY_val = np.loadtxt(validation_path+'/rwY.txt')
                rwZ_val = np.loadtxt(validation_path+'/rwZ.txt')
                laccX_val = np.loadtxt(validation_path+'/laccX.txt')
                laccY_val = np.loadtxt(validation_path+'/laccY.txt')
                laccZ_val = np.loadtxt(validation_path+'/laccZ.txt')
                dlen = min(dlen,np.size(laccZ_val))
                raccX_val = np.loadtxt(validation_path+'/raccX.txt')
                raccY_val = np.loadtxt(validation_path+'/raccY.txt')
                raccZ_val = np.loadtxt(validation_path+'/raccZ.txt')
                dlen = min(dlen,np.size(raccZ_val))

            if(self.useLabels):
                baccX_LL_val = np.loadtxt(validation_path+'/baccX_LL.txt')
                baccY_LL_val = np.loadtxt(validation_path+'/baccY_LL.txt')
                baccZ_LL_val = np.loadtxt(validation_path+'/baccZ_LL.txt')
                baccX_RL_val = np.loadtxt(validation_path+'/baccX_RL.txt')
                baccY_RL_val = np.loadtxt(validation_path+'/baccY_RL.txt')
                baccZ_RL_val = np.loadtxt(validation_path+'/baccZ_RL.txt')
                baccX_val = np.loadtxt(validation_path+'/baccX.txt')
                baccY_val = np.loadtxt(validation_path+'/baccY.txt')
                baccZ_val = np.loadtxt(validation_path+'/baccZ.txt')

                bgX_LL_val = np.loadtxt(training_path+'/bgX_LL.txt')
                bgY_LL_val = np.loadtxt(training_path+'/bgY_LL.txt')
                bgZ_LL_val = np.loadtxt(training_path+'/bgZ_LL.txt')
                bgX_RL_val = np.loadtxt(training_path+'/bgX_RL.txt')
                bgY_RL_val = np.loadtxt(training_path+'/bgY_RL.txt')
                bgZ_RL_val = np.loadtxt(training_path+'/bgZ_RL.txt')
                bgX_val = np.loadtxt(training_path+'/bgX.txt')
                bgY_val = np.loadtxt(training_path+'/bgY.txt')
                bgZ_val = np.loadtxt(training_path+'/bgZ.txt')


                dlen = min(dlen,min(np.size(bgZ_LL_val),np.size(baccZ_RL_val)))
            


        self.data_label = np.array([])
        self.data_train = np.array([])
        self.data_val = np.array([])
        self.data_val_label = np.array([])

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
        if(self.gem2):
            #Leg Linear and Angular Velocities
            self.data_train = np.column_stack([self.data_train, lvX[0:dlen] - rvX[0:dlen]])
            self.data_train = np.column_stack([self.data_train, lvY[0:dlen] - rvY[0:dlen]])
            self.data_train = np.column_stack([self.data_train, lvZ[0:dlen] - rvZ[0:dlen]])
            self.data_train = np.column_stack([self.data_train, lwX[0:dlen] - rwX[0:dlen]])
            self.data_train = np.column_stack([self.data_train, lwY[0:dlen] - rwY[0:dlen]])
            self.data_train = np.column_stack([self.data_train, lwZ[0:dlen] - rwZ[0:dlen]])
            #Leg Linear Acceleration
            self.data_train = np.column_stack([self.data_train, laccX[0:dlen] - raccX[0:dlen]])
            self.data_train = np.column_stack([self.data_train, laccY[0:dlen] - raccY[0:dlen]])
            self.data_train = np.column_stack([self.data_train, laccZ[0:dlen] - raccZ[0:dlen]])

        #Base/Legs Acceleration as labels
        if(self.useLabels):
            self.data_label = bgX_LL[0:dlen]
            self.data_label = np.column_stack([self.data_label, bgY_LL[0:dlen]])
            self.data_label = np.column_stack([self.data_label, bgZ_LL[0:dlen]])
            self.data_label = np.column_stack([self.data_label, bgX_RL[0:dlen]])
            self.data_label = np.column_stack([self.data_label, bgY_RL[0:dlen]])
            self.data_label = np.column_stack([self.data_label, bgZ_RL[0:dlen]])
            self.data_label = np.column_stack([self.data_label, bgX[0:dlen]])
            self.data_label = np.column_stack([self.data_label, bgY[0:dlen]])
            self.data_label = np.column_stack([self.data_label, bgZ[0:dlen]])


            self.data_label = np.column_stack([self.data_label, baccX_LL[0:dlen]])
            self.data_label = np.column_stack([self.data_label, baccY_LL[0:dlen]])
            self.data_label = np.column_stack([self.data_label, baccZ_LL[0:dlen]])
            self.data_label = np.column_stack([self.data_label, baccX_RL[0:dlen]])
            self.data_label = np.column_stack([self.data_label, baccY_RL[0:dlen]])
            self.data_label = np.column_stack([self.data_label, baccZ_RL[0:dlen]])
            self.data_label = np.column_stack([self.data_label, baccX[0:dlen]])
            self.data_label = np.column_stack([self.data_label, baccY[0:dlen]])
            self.data_label = np.column_stack([self.data_label, baccZ[0:dlen]])
            self.data_label_min = np.zeros((self.data_label.shape[1]))
            self.data_label_max = np.zeros((self.data_label.shape[1]))
            self.data_label_mean = np.zeros((self.data_label.shape[1]))
            self.data_label_std = np.zeros((self.data_label.shape[1]))
            #Label Statistics
            for i in range(self.data_label.shape[1]):
                self.data_label_min[i] = np.min(self.data_label[:, i])
                self.data_label_max[i] = np.max(self.data_label[:, i])
                self.data_label_mean[i] = np.mean(self.data_label[:, i])
                self.data_label_std[i] = np.std(self.data_label[:, i])
                #self.data_label[:, i] = self.normalize_data(self.data_label[:, i],self.data_label_max[i], self.data_label_min[i])   
                #self.data_label[:, i] = self.standarize_data(self.data_label[:, i],self.data_label_mean[i], self.data_label_std[i])
                #self.data_label[:, i] = self.normalizeMean_data(self.data_label[:, i],self.data_label_max[i], self.data_label_min[i],self.data_label_mean[i])   





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
            self.data_val = lfX_val[0:dlen] - rfX_val[0:dlen]
            self.data_val = np.column_stack([self.data_val, lfY_val[0:dlen] - rfY_val[0:dlen]])
            self.data_val = np.column_stack([self.data_val, lfZ_val[0:dlen] - rfZ_val[0:dlen]])
            self.data_val = np.column_stack([self.data_val, ltX_val[0:dlen] - rtX_val[0:dlen]])
            self.data_val = np.column_stack([self.data_val, ltY_val[0:dlen] - rtY_val[0:dlen]])
            self.data_val = np.column_stack([self.data_val, ltZ_val[0:dlen] - rtZ_val[0:dlen]])

            #CoM Velocity
            self.data_val = np.column_stack([self.data_val, dcX_val[0:dlen]])
            self.data_val = np.column_stack([self.data_val, dcY_val[0:dlen]])
            self.data_val = np.column_stack([self.data_val, dcZ_val[0:dlen]])
            if(self.gem2):
                #Leg Linear and Angular Velocities
                self.data_val = np.column_stack([self.data_val, lvX_val[0:dlen] - rvX_val[0:dlen]])
                self.data_val = np.column_stack([self.data_val, lvY_val[0:dlen] - rvY_val[0:dlen]])
                self.data_val = np.column_stack([self.data_val, lvZ_val[0:dlen] - rvZ_val[0:dlen]])
                self.data_val = np.column_stack([self.data_val, lwX_val[0:dlen] - rwX_val[0:dlen]])
                self.data_val = np.column_stack([self.data_val, lwY_val[0:dlen] - rwY_val[0:dlen]])
                self.data_val = np.column_stack([self.data_val, lwZ_val[0:dlen] - rwZ_val[0:dlen]])
                #Leg Linear Acceleration
                self.data_val = np.column_stack([self.data_val, laccX_val[0:dlen] - raccX_val[0:dlen]])
                self.data_val = np.column_stack([self.data_val, laccY_val[0:dlen] - raccY_val[0:dlen]])
                self.data_val = np.column_stack([self.data_val, laccZ_val[0:dlen] - raccZ_val[0:dlen]])


            #Base/Legs Acceleration as labels
            if(self.useLabels):
                self.data_val_label = bgX_LL_val[0:dlen]    
                self.data_val_label = np.column_stack([self.data_val_label, bgY_LL_val[0:dlen]])
                self.data_val_label = np.column_stack([self.data_val_label, bgZ_LL_val[0:dlen]])
                self.data_val_label = np.column_stack([self.data_val_label, bgX_RL_val[0:dlen]])
                self.data_val_label = np.column_stack([self.data_val_label, bgY_RL_val[0:dlen]])
                self.data_val_label = np.column_stack([self.data_val_label, bgZ_RL_val[0:dlen]])
                self.data_val_label = np.column_stack([self.data_val_label, bgX_val[0:dlen]])
                self.data_val_label = np.column_stack([self.data_val_label, bgY_val[0:dlen]])
                self.data_val_label = np.column_stack([self.data_val_label, bgZ_val[0:dlen]])


                self.data_val_label = np.column_stack([self.data_val_label, baccX_LL_val[0:dlen]])
                self.data_val_label = np.column_stack([self.data_val_label, baccY_LL_val[0:dlen]])
                self.data_val_label = np.column_stack([self.data_val_label, baccZ_LL_val[0:dlen]])
                self.data_val_label = np.column_stack([self.data_val_label, baccX_RL_val[0:dlen]])
                self.data_val_label = np.column_stack([self.data_val_label, baccY_RL_val[0:dlen]])
                self.data_val_label = np.column_stack([self.data_val_label, baccZ_RL_val[0:dlen]])
                self.data_val_label = np.column_stack([self.data_val_label, baccX_val[0:dlen]])
                self.data_val_label = np.column_stack([self.data_val_label, baccY_val[0:dlen]])
                self.data_val_label = np.column_stack([self.data_val_label, baccZ_val[0:dlen]])

                #Validation Label Statistics
                self.data_val_label_min = np.zeros((self.data_val_label.shape[1]))
                self.data_val_label_max = np.zeros((self.data_val_label.shape[1]))
                self.data_val_label_mean = np.zeros((self.data_val_label.shape[1]))
                self.data_val_label_std = np.zeros((self.data_val_label.shape[1]))
                for i in range(self.data_val_label.shape[1]):
                    self.data_val_label_min[i] = np.min(self.data_val_label[:, i])
                    self.data_val_label_max[i] = np.max(self.data_val_label[:, i])
                    self.data_val_label_mean[i] = np.mean(self.data_val_label[:, i])
                    self.data_val_label_std[i] = np.std(self.data_val_label[:, i])
                    #self.data_val_label[:, i] = self.normalize_data(self.data_val_label[:, i],self.data_val_label_max[i], self.data_val_label_min[i])   
                    #self.data_val_label[:, i] = self.standarize_data(self.data_val_label[:, i],self.data_val_label_mean[i], self.data_val_label_std[i])
                    #self.data_val_label[:, i] = self.normalizeMean_data(self.data_val_label[:, i],self.data_val_label_max[i], self.data_val_label_min[i],self.data_val_label_mean[i])


            #Base Linear Acceleration and Base Angular Velocity
            self.data_val = np.column_stack([self.data_val, accX_val[0:dlen]])
            self.data_val = np.column_stack([self.data_val, accY_val[0:dlen]])
            self.data_val = np.column_stack([self.data_val, accZ_val[0:dlen]])
            self.data_val = np.column_stack([self.data_val, gX_val[0:dlen]])
            self.data_val = np.column_stack([self.data_val, gY_val[0:dlen]])
            self.data_val = np.column_stack([self.data_val, gZ_val[0:dlen]])



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

        if(not gt.gem2):
            output_ = np.append(output_, gt.cXdt.diff(data.cX), axis = 0)
            output_ = np.append(output_, gt.cYdt.diff(data.cY), axis = 0)
            output_ = np.append(output_, gt.cZdt.diff(data.cZ), axis = 0)
            output_ = np.append(output_, data.accX, axis = 0)
            output_ = np.append(output_, data.accY, axis = 0)
            output_ = np.append(output_, data.accZ, axis = 0)
            output_ = np.append(output_, data.gX, axis = 0)
            output_ = np.append(output_, data.gY, axis = 0)
            output_ = np.append(output_, data.gZ, axis = 0)
        else:
            output_ = np.append(output_, data.dcX, axis = 0)
            output_ = np.append(output_, data.dcY, axis = 0)
            output_ = np.append(output_, data.dcZ, axis = 0)
            output_ = np.append(output_, data.lvX - data.rvX, axis = 0)
            output_ = np.append(output_, data.lvY - data.rvY, axis = 0)
            output_ = np.append(output_, data.lvZ - data.rvZ, axis = 0)
            output_ = np.append(output_, data.lwX - data.rwX, axis = 0)
            output_ = np.append(output_, data.lwY - data.rwY, axis = 0)
            output_ = np.append(output_, data.lwZ - data.rwZ, axis = 0)

            if(not gt.useLabels):
                output_ = np.append(output_, data.laccX - data.raccX, axis = 0)
                output_ = np.append(output_, data.laccY - data.raccY, axis = 0)
                output_ = np.append(output_, data.laccZ - data.raccZ, axis = 0)
            else:
                output_ = np.append(output_, data.baccX_LL, axis = 0)
                output_ = np.append(output_, data.baccY_LL, axis = 0)
                output_ = np.append(output_, data.baccZ_LL, axis = 0)
                output_ = np.append(output_, data.baccX_RL, axis = 0)
                output_ = np.append(output_, data.baccY_RL, axis = 0)
                output_ = np.append(output_, data.baccZ_RL, axis = 0)
                output_ = np.append(output_, data.baccX, axis = 0)
                output_ = np.append(output_, data.baccY, axis = 0)
                output_ = np.append(output_, data.baccZ, axis = 0)

        for i in range(self.data_train.shape[1]):
            output_[i] = self.normalize_data(output_[i],self.data_train_max[i], self.data_train_min[i])   
            #output_[i] = self.normalizeMean_data(output_[i],self.data_train_max[i], self.data_train_min[i], self.data_train_mean[i])   


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



    def plot_accelerations_LR(self,leg_probabilities, data_labels):
        t = np.arange(0,leg_probabilities.shape[0], 1)
        base_accX = data_labels[:,15]
        base_accY = data_labels[:,16]
        base_accZ = data_labels[:,17]
        
        est_accX = np.divide( (np.multiply(leg_probabilities[:,0], data_labels[:,9]) + np.multiply(leg_probabilities[:,1],data_labels[:,12]) ), (leg_probabilities[:,0] + leg_probabilities[:,1]))
        est_accY = np.divide( (np.multiply(leg_probabilities[:,0], data_labels[:,10]) + np.multiply(leg_probabilities[:,1],data_labels[:,13])), (leg_probabilities[:,0] + leg_probabilities[:,1]))
        est_accZ = np.divide( (np.multiply(leg_probabilities[:,0], data_labels[:,11]) + np.multiply(leg_probabilities[:,1],data_labels[:,14])), (leg_probabilities[:,0] + leg_probabilities[:,1]))

        est_accX_LL = np.ma.masked_where(leg_probabilities[:,0] < leg_probabilities[:,1], est_accX)
        est_accX_RL = np.ma.masked_where(leg_probabilities[:,0] >= leg_probabilities[:,1], est_accX)
        est_accY_LL = np.ma.masked_where(leg_probabilities[:,0] < leg_probabilities[:,1], est_accY)
        est_accY_RL = np.ma.masked_where(leg_probabilities[:,0] >= leg_probabilities[:,1], est_accY)
        est_accZ_LL = np.ma.masked_where(leg_probabilities[:,0] < leg_probabilities[:,1], est_accZ)
        est_accZ_RL = np.ma.masked_where(leg_probabilities[:,0] >= leg_probabilities[:,1], est_accZ)

        plt.figure()
        plt.subplot(311)
        plt.plot(t,base_accX)
        #plt.scatter(t,est_accX,s=2.0, color=c)
        plt.plot(t,est_accX_LL,t,est_accX_RL)
        plt.ylabel('$acc_x$')
        plt.subplot(312)
        plt.plot(t,base_accY)
        #plt.scatter(t,est_accY, s=2.0,color=c)
        plt.plot(t,est_accY_LL,t,est_accY_RL)
        plt.ylabel('$acc_y$')
        plt.subplot(313)
        plt.plot(t,base_accZ)
        #plt.scatter(t,est_accZ, s=2.0,color=c)
        plt.plot(t,est_accZ_LL,t,est_accZ_RL)
        plt.ylabel('$acc_z$')
        plt.xlabel('$samples$')
        plt.show()

        base_gX = data_labels[:,6]
        base_gY = data_labels[:,7]
        base_gZ = data_labels[:,8]
        
        est_gX = np.divide( (np.multiply(leg_probabilities[:,0], data_labels[:,0]) + np.multiply(leg_probabilities[:,1],data_labels[:,3])), (leg_probabilities[:,0] + leg_probabilities[:,1]))
        est_gY = np.divide( (np.multiply(leg_probabilities[:,0], data_labels[:,1]) + np.multiply(leg_probabilities[:,1],data_labels[:,4])), (leg_probabilities[:,0] + leg_probabilities[:,1]))
        est_gZ = np.divide( (np.multiply(leg_probabilities[:,0], data_labels[:,2]) + np.multiply(leg_probabilities[:,1],data_labels[:,5])), (leg_probabilities[:,0] + leg_probabilities[:,1]))

        est_gX_LL = np.ma.masked_where(leg_probabilities[:,0] < leg_probabilities[:,1], est_gX)
        est_gX_RL = np.ma.masked_where(leg_probabilities[:,0] >= leg_probabilities[:,1], est_gX)
        est_gY_LL = np.ma.masked_where(leg_probabilities[:,0] < leg_probabilities[:,1], est_gY)
        est_gY_RL = np.ma.masked_where(leg_probabilities[:,0] >= leg_probabilities[:,1], est_gY)
        est_gZ_LL = np.ma.masked_where(leg_probabilities[:,0] < leg_probabilities[:,1], est_gZ)
        est_gZ_RL = np.ma.masked_where(leg_probabilities[:,0] >= leg_probabilities[:,1], est_gZ)

        plt.figure()
        plt.subplot(311)
        plt.plot(t,base_gX)
        #plt.scatter(t,est_gX,s=2.0, color=c)
        plt.plot(t,est_gX_LL,t,est_gX_RL)
        plt.ylabel('$g_x$')
        plt.subplot(312)
        plt.plot(t,base_gY)
        #plt.scatter(t,est_gY, s=2.0,color=c)
        plt.plot(t,est_gY_LL,t,est_gY_RL)
        plt.ylabel('$g_y$')
        plt.subplot(313)
        plt.plot(t,base_gZ)
        #plt.scatter(t,est_gZ, s=2.0,color=c)
        plt.plot(t,est_gZ_LL,t,est_gZ_RL)
        plt.ylabel('$g_z$')
        plt.xlabel('$samples$')
        plt.show()




    def plot_accelerations_LRD(self,leg_probabilities, data_labels, predicted_labels):
        t = np.arange(0,leg_probabilities.shape[0], 1)
        base_accX = data_labels[:,6]
        base_accY = data_labels[:,7]
        base_accZ = data_labels[:,8]
        est_accX = leg_probabilities[:,0].T*data_labels[:,0] + leg_probabilities[:,1].T*data_labels[:,3]
        est_accY = leg_probabilities[:,0].T*data_labels[:,1] + leg_probabilities[:,1].T*data_labels[:,4]
        est_accZ = leg_probabilities[:,0].T*data_labels[:,2] + leg_probabilities[:,1].T*data_labels[:,5]
        est_accX_LSS = np.ma.masked_where(predicted_labels == 0, est_accX)
        est_accX_RSS = np.ma.masked_where(predicted_labels == 1, est_accX)
        est_accX_DS = np.ma.masked_where(predicted_labels == 2, est_accX)
        est_accY_LSS = np.ma.masked_where(predicted_labels == 0, est_accY)
        est_accY_RSS =np.ma.masked_where(predicted_labels == 1, est_accY)
        est_accY_DS = np.ma.masked_where(predicted_labels == 2, est_accY)
        est_accZ_LSS = np.ma.masked_where(predicted_labels == 0, est_accZ)
        est_accZ_RSS = np.ma.masked_where(predicted_labels == 1, est_accZ)
        est_accZ_DS = np.ma.masked_where(predicted_labels == 2, est_accZ)

        plt.figure()
        plt.subplot(311)
        plt.plot(t,base_accX)
        plt.plot(t,est_accX_LSS,t,est_accX_RSS, t,est_accX_DS)
        plt.grid('on')
        plt.ylabel('$acc_x$')

        plt.subplot(312)
        plt.plot(t,base_accY)
        plt.plot(t,est_accY_LSS,t,est_accY_RSS, t,est_accY_DS)
        plt.grid('on')
        plt.ylabel('$acc_y$')

        plt.subplot(313)
        plt.plot(t,base_accZ)
        plt.plot(t,est_accZ_LSS,t,est_accZ_RSS, t,est_accZ_DS)
        plt.ylabel('$acc_z$')
        plt.xlabel('$samples$')
        plt.grid('on')
        plt.show()


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
        #pdf=PdfPages(title+".pdf")
        #pdf.savefig(fig)


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


class diff_tool():
    def __init__(self, x_=None, dx_=None):
        self.x = x_
        self.dx = dx_

    def diff(self,x_):
        if(self.x is None):
            self.dx = 0
        else:
            self.dx = x_ - self.x

        self.x = x_
        return self.dx

    def reset(self):
        self.dx = None
        self.x = None


class cf:
    def __init__(self,freq_ = 100.0, alpha_ = 0.98):
        self.alpha = alpha_
        self.freq = freq_
        self.roll = 0.0
        self.pitch = 0.0
        self.firstrun = True


    def computeAccAngle(self,accX,accY,accZ):
        roll = atan2(accY,sqrt(accX*accX+accZ*accZ))
        pitch = atan2(accX,sqrt(accZ*accZ+accY*accY))
        return roll,pitch

    def update(self,accX,accY,accZ,gX,gY):
        roll_, pitch_ = self.computeAccAngle(accX,accY,accZ)
        if(self.firstrun):
            self.roll =  roll_
            self.pitch = pitch_
            self.firstrun = False
        else:
            self.roll = self.alpha * (self.roll + gX * 1.0/self.freq) +  (1.0 - self.alpha)*roll_
            self.pitch = self.alpha * (self.pitch + gY * 1.0/self.freq) +  (1.0 - self.alpha)*pitch_
        return self.roll, self.pitch


    def reset(self):
        self.roll = 0.0
        self.pitch = 0.0
        self.firstrun = True
