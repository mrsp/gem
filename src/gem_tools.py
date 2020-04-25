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
 *	 nor the names of its contributors may be used to endorse or promote products derived from
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
    'text.usetex': False,
    'figure.figsize': [7, 4] # instead of 4.5, 4.5
}
plt.rcParams.update(params)

class GeM_tools():
    def __init__(self, gt_comparison=False):
        self.gt_comparison = gt_comparison
        self.cXdt = diff_tool()
        self.cYdt = diff_tool()
        self.cZdt = diff_tool()


    def input_data(self, setpath):

        cX = np.loadtxt(setpath+'/c_encx.txt')
        cY = np.loadtxt(setpath+'/c_ency.txt')
        cZ = np.loadtxt(setpath+'/c_encz.txt')

        rfX = np.loadtxt(setpath+'/rfX.txt')
        rfY = np.loadtxt(setpath+'/rfY.txt')
        rfZ = np.loadtxt(setpath+'/rfZ.txt')
        rtX = np.loadtxt(setpath+'/rtX.txt')
        rtY = np.loadtxt(setpath+'/rtY.txt')
        rtZ = np.loadtxt(setpath+'/rtZ.txt')


        lfX = np.loadtxt(setpath+'/lfX.txt')
        lfY = np.loadtxt(setpath+'/lfY.txt')
        lfZ = np.loadtxt(setpath+'/lfZ.txt')
        ltX = np.loadtxt(setpath+'/ltX.txt')
        ltY = np.loadtxt(setpath+'/ltY.txt')
        ltZ = np.loadtxt(setpath+'/ltZ.txt')
        

        
        lvX = np.loadtxt(setpath+'/accX.txt')
        lvY = np.loadtxt(setpath+'/accX.txt')
        lvZ = np.loadtxt(setpath+'/accX.txt')
        lwX = np.loadtxt(setpath+'/accX.txt')
        lwY = np.loadtxt(setpath+'/accX.txt')
        lwZ = np.loadtxt(setpath+'/accX.txt')
        
        rvX = np.loadtxt(setpath+'/gX.txt')
        rvY = np.loadtxt(setpath+'/gX.txt')
        rvZ = np.loadtxt(setpath+'/gX.txt')
        rwX = np.loadtxt(setpath+'/gX.txt')
        rwY = np.loadtxt(setpath+'/gX.txt')
        rwZ = np.loadtxt(setpath+'/gX.txt')


        gX = np.loadtxt(setpath+'/gX.txt')
        gY = np.loadtxt(setpath+'/gY.txt')
        gZ = np.loadtxt(setpath+'/gZ.txt')
        accX = np.loadtxt(setpath+'/accX.txt')
        accY = np.loadtxt(setpath+'/accY.txt')
        accZ = np.loadtxt(setpath+'/accZ.txt')

        
        if(self.gt_comparison):
            gt_lfZ  = np.loadtxt(setpath+'/gt_lfZ.txt')
            gt_rfZ  = np.loadtxt(setpath+'/gt_rfZ.txt')
            gt_lfX  = np.loadtxt(setpath+'/gt_lfX.txt')
            gt_rfX  = np.loadtxt(setpath+'/gt_rfX.txt')
            gt_lfY  = np.loadtxt(setpath+'/gt_lfY.txt')
            gt_rfY  = np.loadtxt(setpath+'/gt_rfY.txt')
            mu  = np.loadtxt(setpath+'/mu.txt')
       	    self.mu = mu
       


        dlen0 = np.size(cX)
        dlen1 = np.size(cY)
        dlen2 = np.size(lfZ)
        dlen3 = np.size(accZ)
        dlen6 = np.size(rfZ)


        if(self.gt_comparison):
            dlen4 = np.size(gt_lfZ)
            dlen5 = np.size(gt_rfX)
            dlen = min(dlen0, dlen1, dlen2, dlen3, dlen4, dlen5,dlen6)
        else:
            dlen = min(dlen0, dlen1, dlen2, dlen3,dlen6)



        dcX = np.zeros((dlen))
        dcY = np.zeros((dlen))
        dcZ= np.zeros((dlen))
     
        if(self.gt_comparison):
            phase = np.zeros((dlen))




        for i in range(dlen):
            dcX[i]=self.cXdt.diff(cX[i])
            dcY[i]=self.cYdt.diff(cY[i])
            dcZ[i]=self.cZdt.diff(cZ[i])

            if(self.gt_comparison):
                lcon = np.sqrt(gt_lfX[i] * gt_lfX[i] + gt_lfY[i] * gt_lfY[i])
                rcon = np.sqrt(gt_rfX[i] * gt_rfX[i] + gt_rfY[i] * gt_rfY[i])
                if( ((self.mu[i]*gt_lfZ[i])>lcon) and ((self.mu[i] * gt_rfZ[i])>rcon)):
                    phase[i] = 2
                elif( (self.mu[i]*gt_lfZ[i])>lcon ):
                    phase[i] = 1
                elif( (self.mu[i]*gt_rfZ[i])>rcon ):
                    phase[i] = 0
                else:
                    phase[i] = -1


        self.data_train = np.zeros((dlen, 21))
        #Leg Forces and Torques
        self.data_train[:, 0] = lfX[0:dlen] - rfX[0:dlen]
        self.data_train[:, 1] = lfY[0:dlen] - rfY[0:dlen]
        self.data_train[:, 2] = lfZ[0:dlen] - rfZ[0:dlen]
        self.data_train[:, 3] = ltX[0:dlen] - rtX[0:dlen]
        self.data_train[:, 4] = ltY[0:dlen] - rtY[0:dlen]
        self.data_train[:, 5] = ltZ[0:dlen] - rtZ[0:dlen]
        #CoM Velocity
        self.data_train[1:dlen, 6] = dcX[1:dlen]
        self.data_train[1:dlen, 7] = dcY[1:dlen]
        self.data_train[1:dlen, 8] = dcZ[1:dlen]
        self.data_train[0, 6] = dcX[1]
        self.data_train[0, 7] = dcY[1]
        self.data_train[0, 8] = dcZ[1]
        #Leg Linear and Angular Velocities
        self.data_train[:, 9]  = lvX[0:dlen] - rvX[0:dlen]
        self.data_train[:, 10] = lvY[0:dlen] - rvY[0:dlen]
        self.data_train[:, 11] = lvZ[0:dlen] - rvZ[0:dlen]
        self.data_train[:, 12] = lwX[0:dlen] - rwX[0:dlen]
        self.data_train[:, 13] = lwY[0:dlen] - rwY[0:dlen]
        self.data_train[:, 14] = lwZ[0:dlen] - rwZ[0:dlen]
        #Base Linear Acceleration and Base Angular Velocity
        self.data_train[:, 15] = accX[0:dlen]
        self.data_train[:, 16] = accY[0:dlen]
        self.data_train[:, 17] = accZ[0:dlen]
        self.data_train[:, 18] = gX[0:dlen]
        self.data_train[:, 19] = gY[0:dlen]
        self.data_train[:, 20] = gZ[0:dlen]

        #Data Statistics
        #fX
        self.dfX_train_min = min(self.data_train[:, 0])
        self.dfX_train_max = max(self.data_train[:, 0])
        self.dfX_train_mean = np.mean(self.data_train[:, 0])
        self.dfX_train_std = np.std(self.data_train[:, 0])
        #fY
        self.dfY_train_min = min(self.data_train[:, 1])
        self.dfY_train_max = max(self.data_train[:, 1])
        self.dfY_train_mean = np.mean(self.data_train[:, 1])
        self.dfY_train_std = np.std(self.data_train[:, 1])
        #fZ
        self.dfZ_train_min = min(self.data_train[:, 2])
        self.dfZ_train_max = max(self.data_train[:, 2])
        self.dfZ_train_mean = np.mean(self.data_train[:, 2])
        self.dfZ_train_std = np.std(self.data_train[:, 2])
        #tX
        self.dtX_train_min = min(self.data_train[:, 3])
        self.dtX_train_max = max(self.data_train[:, 3])
        self.dtX_train_mean = np.mean(self.data_train[:, 3])
        self.dtX_train_std = np.std(self.data_train[:, 3])
        #tY
        self.dtY_train_min = min(self.data_train[:, 4])
        self.dtY_train_max = max(self.data_train[:, 4])
        self.dtY_train_mean = np.mean(self.data_train[:, 4])
        self.dtY_train_std = np.std(self.data_train[:, 4])
        #tZ
        self.dtZ_train_min = min(self.data_train[:, 5])
        self.dtZ_train_max = max(self.data_train[:, 5])
        self.dtZ_train_mean = np.mean(self.data_train[:, 5])
        self.dtZ_train_std = np.std(self.data_train[:, 5])
        #cX
        self.dcX_train_min = min(self.data_train[:, 6])
        self.dcX_train_max = max(self.data_train[:, 6])
        self.dcX_train_mean = np.mean(self.data_train[:, 6])
        self.dcX_train_std = np.std(self.data_train[:, 6])
        #cY
        self.dcY_train_min = min(self.data_train[:, 7])
        self.dcY_train_max = max(self.data_train[:, 7])
        self.dcY_train_mean = np.mean(self.data_train[:, 7])
        self.dcY_train_std = np.std(self.data_train[:, 7])
        #cZ
        self.dcZ_train_min = min(self.data_train[:, 8])
        self.dcZ_train_max = max(self.data_train[:, 8])
        self.dcZ_train_mean = np.mean(self.data_train[:, 8])
        self.dcZ_train_std = np.std(self.data_train[:, 8])
        #vX
        self.dvX_train_min = min(self.data_train[:, 9])
        self.dvX_train_max = max(self.data_train[:, 9])
        self.dvX_train_mean = np.mean(self.data_train[:, 9])
        self.dvX_train_std = np.std(self.data_train[:, 9])
        #vY
        self.dvY_train_min = min(self.data_train[:, 10])
        self.dvY_train_max = max(self.data_train[:, 10])
        self.dvY_train_mean = np.mean(self.data_train[:, 10])
        self.dvY_train_std = np.std(self.data_train[:, 10])
        #vZ
        self.dvZ_train_min = min(self.data_train[:, 11])
        self.dvZ_train_max = max(self.data_train[:, 11])
        self.dvZ_train_mean = np.mean(self.data_train[:, 11])
        self.dvZ_train_std = np.std(self.data_train[:, 11])
        #wX
        self.dwX_train_min = min(self.data_train[:, 12])    
        self.dwX_train_max = max(self.data_train[:, 12])
        self.dwX_train_mean = np.mean(self.data_train[:, 12])
        self.dwX_train_std = np.std(self.data_train[:, 12])
        #wY
        self.dwY_train_min = min(self.data_train[:, 13])
        self.dwY_train_max = max(self.data_train[:, 13])
        self.dwY_train_mean = np.mean(self.data_train[:, 13])
        self.dwY_train_std = np.std(self.data_train[:, 13])
        #wZ
        self.dwZ_train_min = min(self.data_train[:, 14])
        self.dwZ_train_max = max(self.data_train[:, 14])
        self.dwZ_train_mean = np.mean(self.data_train[:, 14])
        self.dwZ_train_std = np.std(self.data_train[:, 14])
        #accX
        self.accX_train_min = min(self.data_train[:, 15])
        self.accX_train_max = max(self.data_train[:, 15])
        self.accX_train_mean = np.mean(self.data_train[:, 15])
        self.accX_train_std = np.std(self.data_train[:, 15])
        #accY
        self.accY_train_min = min(self.data_train[:, 16])
        self.accY_train_max = max(self.data_train[:, 16])
        self.accY_train_mean = np.mean(self.data_train[:, 16])
        self.accY_train_std = np.std(self.data_train[:, 16])
        #accZ
        self.accZ_train_min = min(self.data_train[:, 17])
        self.accZ_train_max = max(self.data_train[:, 17])
        self.accZ_train_mean = np.mean(self.data_train[:, 17])
        self.accZ_train_std = np.std(self.data_train[:, 17])
        #gX
        self.gX_train_min = min(self.data_train[:, 18])
        self.gX_train_max = max(self.data_train[:, 18])
        self.gX_train_mean = np.mean(self.data_train[:, 18])
        self.gX_train_std = np.std(self.data_train[:, 18])
        #gY
        self.gY_train_min = min(self.data_train[:, 19])
        self.gY_train_max = max(self.data_train[:, 19])
        self.gY_train_mean = np.mean(self.data_train[:, 19])
        self.gY_train_std = np.std(self.data_train[:, 19])
        #gZ
        self.gZ_train_min = min(self.data_train[:, 20])
        self.gZ_train_max = max(self.data_train[:, 20])
        self.gZ_train_mean = np.mean(self.data_train[:, 20])
        self.gZ_train_std = np.std(self.data_train[:, 20])

        #Normalization or Standarization?
        self.data_train[:, 0] = self.normalize_data(self.data_train[:, 0],self.dfX_train_max, self.dfX_train_min)   
        self.data_train[:, 1] = self.normalize_data(self.data_train[:, 1],self.dfY_train_max, self.dfY_train_min)   
        self.data_train[:, 2] = self.normalize_data(self.data_train[:, 2],self.dfZ_train_max, self.dfZ_train_min)   
        self.data_train[:, 3] = self.normalize_data(self.data_train[:, 3],self.dtX_train_max, self.dtX_train_min)   
        self.data_train[:, 4] = self.normalize_data(self.data_train[:, 4],self.dtY_train_max, self.dtY_train_min)   
        self.data_train[:, 5] = self.normalize_data(self.data_train[:, 5],self.dtZ_train_max, self.dtZ_train_min)   
        self.data_train[:, 6] = self.normalize_data(self.data_train[:, 6],self.dcX_train_max, self.dcX_train_min)   
        self.data_train[:, 7] = self.normalize_data(self.data_train[:, 7],self.dcY_train_max, self.dcY_train_min)   
        self.data_train[:, 8] = self.normalize_data(self.data_train[:, 8],self.dcZ_train_max, self.dcZ_train_min) 
        self.data_train[:, 9] = self.normalize_data(self.data_train[:, 9],self.dvX_train_max, self.dvX_train_min) 
        self.data_train[:, 10] = self.normalize_data(self.data_train[:, 10],self.dvY_train_max, self.dvY_train_min) 
        self.data_train[:, 11] = self.normalize_data(self.data_train[:, 11],self.dvZ_train_max, self.dvZ_train_min) 
        self.data_train[:, 12] = self.normalize_data(self.data_train[:, 12],self.dwX_train_max, self.dwX_train_min) 
        self.data_train[:, 13] = self.normalize_data(self.data_train[:, 13],self.dwY_train_max, self.dwY_train_min) 
        self.data_train[:, 14] = self.normalize_data(self.data_train[:, 14],self.dwZ_train_max, self.dwZ_train_min) 
        self.data_train[:, 15] = self.normalize_data(self.data_train[:, 15],self.accX_train_max, self.accX_train_min) 
        self.data_train[:, 16] = self.normalize_data(self.data_train[:, 16],self.accY_train_max, self.accY_train_min) 
        self.data_train[:, 17] = self.normalize_data(self.data_train[:, 17],self.accZ_train_max, self.accZ_train_min) 
        self.data_train[:, 18] = self.normalize_data(self.data_train[:, 18],self.gX_train_max, self.gX_train_min) 
        self.data_train[:, 19] = self.normalize_data(self.data_train[:, 19],self.gY_train_max, self.gY_train_min) 
        self.data_train[:, 20] = self.normalize_data(self.data_train[:, 20],self.gZ_train_max, self.gZ_train_min) 

        '''
        self.data_train[:, 0] = self.standarize_data(self.data_train[:, 0],self.dfX_train_mean, self.dfX_train_std)   
        self.data_train[:, 1] = self.standarize_data(self.data_train[:, 1],self.dfY_train_mean, self.dfY_train_std)   
        self.data_train[:, 2] = self.standarize_data(self.data_train[:, 2],self.dfZ_train_mean, self.dfZ_train_std)   
        self.data_train[:, 3] = self.standarize_data(self.data_train[:, 3],self.dtX_train_mean, self.dtX_train_std)   
        self.data_train[:, 4] = self.standarize_data(self.data_train[:, 4],self.dtY_train_mean, self.dtY_train_std)   
        self.data_train[:, 5] = self.standarize_data(self.data_train[:, 5],self.dtZ_train_mean, self.dtZ_train_std)   
        self.data_train[:, 6] = self.standarize_data(self.data_train[:, 6],self.dcX_train_mean, self.dcX_train_std)   
        self.data_train[:, 7] = self.standarize_data(self.data_train[:, 7],self.dcY_train_mean, self.dcY_train_std)   
        self.data_train[:, 8] = self.standarize_data(self.data_train[:, 8],self.dcZ_train_mean, self.dcZ_train_std)   
        self.data_train[:, 9] = self.standarize_data(self.data_train[:, 9],self.droll_train_mean, self.droll_train_std)   
        self.data_train[:, 10] = self.standarize_data(self.data_train[:, 10],self.dpitch_train_mean, self.dpitch_train_std)   
        '''


        if (self.gt_comparison):
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
        else:
            self.dlen = dlen




        self.cXdt.reset()
        self.cYdt.reset()
        self.cZdt.reset()




    def genInput(self,cX,cY,cZ,accX,accY,accZ,gX,gY,gZ,lvX,lvY,lvZ,rvX,rvY,rvZ,lwX,lwY,lwZ,rwX,rwY,rwZ,lfX,lfY,lfZ,rfX,rfY,rfZ,ltX,ltY,ltZ,rtX,rtY,rtZ ,gt=None):

        if gt is None:
            gt=self

        output_ = np.zeros(21)
        dcX = gt.cXdt.diff(cX)
        dcY = gt.cYdt.diff(cY)
        dcZ = gt.cZdt.diff(cZ)
        dfZ = lfZ - rfZ
        dfX = lfX - rfX
        dfY = lfY - rfY
        dtZ = ltZ - rtZ
        dtX = ltX - rtX
        dtY = ltY - rtY
        
        dfX = self.normalize(dfX, gt.dfX_train_max, gt.dfX_train_min)  
        dfY = self.normalize(dfY, gt.dfY_train_max, gt.dfY_train_min)  
        dfZ = self.normalize(dfZ, gt.dfZ_train_max, gt.dfZ_train_min)  
        
        dtX = self.normalize(dtX, gt.dtX_train_max, gt.dtX_train_min)  
        dtY = self.normalize(dtY, gt.dtY_train_max, gt.dtY_train_min)  
        dtZ = self.normalize(dtZ, gt.dtZ_train_max, gt.dtZ_train_min) 
        
        dcX = self.normalize(dcX, gt.dcX_train_max, gt.dcX_train_min)  
        dcY = self.normalize(dcY, gt.dcY_train_max, gt.dcY_train_min)  
        dcZ = self.normalize(dcZ, gt.dcZ_train_max, gt.dcZ_train_min) 


        dvX = self.normalize(dvX, gt.dvX_train_max, gt.dvX_train_min)  
        dvY = self.normalize(dvY, gt.dvY_train_max, gt.dvY_train_min)  
        dvZ = self.normalize(dvZ, gt.dvZ_train_max, gt.dvZ_train_min)  

        dwX = self.normalize(dwX, gt.dwX_train_max, gt.dwX_train_min)  
        dwY = self.normalize(dwY, gt.dwY_train_max, gt.dwY_train_min)  
        dwZ = self.normalize(dwZ, gt.dwZ_train_max, gt.dwZ_train_min)  


        accX = self.normalize(accX, gt.accX_train_max, gt.accX_train_min)  
        accY = self.normalize(accY, gt.accY_train_max, gt.accY_train_min)  
        accZ = self.normalize(accZ, gt.accZ_train_max, gt.accZ_train_min) 


        gX = self.normalize(gX, gt.gX_train_max, gt.gX_train_min)  
        gY = self.normalize(gY, gt.gY_train_max, gt.gY_train_min)  
        gZ = self.normalize(gZ, gt.gZ_train_max, gt.gZ_train_min) 


        output_[0] = dfX
        output_[1] = dfY
        output_[2] = dfZ
        output_[3] = dtX
        output_[4] = dtY
        output_[5] = dtZ
        output_[6] = dcX
        output_[7] = dcY
        output_[8] = dcZ
        output_[9] = dvX
        output_[10] = dvY
        output_[11] = dvZ
        output_[12] = dwX
        output_[13] = dwY
        output_[14] = dwZ
        output_[15] = accX
        output_[16] = accY
        output_[17] = accZ
        output_[18] = gX
        output_[19] = gY
        output_[20] = gZ

        return output_


    def normalize_data(self,din, dmax, dmin):    
        if(dmax-dmin != 0):
            dout = (din - dmin)/(dmax-dmin)
        else:
            dout =  np.zeros((np.size(din)))

        return dout

    def standarize_data(self,din,dmean,dstd):
        if(dstd != 0):
            dout = (din - dmean)/dstd
        else:
            dout =  np.zeros((np.size(din)))

        return dout


    def normalize(self,din, dmax, dmin):    
        if(din>dmax):
            din=dmax
        elif(din<dmin):
            din=dmin

        if(dmax-dmin != 0):
            dout = (din - dmin)/(dmax-dmin)
        else:
            dout =  0

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

            mean=np.zeros((3,2))
            mean[0,0]=np.mean(d1[:,0])
            mean[0,1]=np.mean(d1[:,1])
            mean[1,0]=np.mean(d2[:,0])

            mean[1,1]=np.mean(d2[:,1])
            mean[2,0]=np.mean(d3[:,0])
            mean[2,1]=np.mean(d3[:,1])

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
