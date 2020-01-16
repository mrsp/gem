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
    def __init__(self, comp_filtering=False, freq=500, a=0.9999, gt_comparison=False):

        self.comp_filtering = comp_filtering
        self.gt_comparison = gt_comparison

        if(comp_filtering):
            self.compf = cf(freq, a)

        self.cXdt = diff_tool()
        self.cYdt = diff_tool()
        self.cZdt = diff_tool()
        self.rolldt = diff_tool()
        self.pitchdt = diff_tool()

    def input_data(self, setpath,blf,blt,brf,brt):

        cX = np.loadtxt(setpath+'/c_encx.txt')
        cY = np.loadtxt(setpath+'/c_ency.txt')
        cZ = np.loadtxt(setpath+'/c_encz.txt')
        lfZ = np.loadtxt(setpath+'/lfZ.txt')
        rfZ = np.loadtxt(setpath+'/rfZ.txt')
        rfX = np.loadtxt(setpath+'/rfX.txt')
        rfY = np.loadtxt(setpath+'/rfY.txt')
        lfX = np.loadtxt(setpath+'/lfX.txt')
        lfY = np.loadtxt(setpath+'/lfY.txt')
        ltX = np.loadtxt(setpath+'/ltX.txt')
        ltY = np.loadtxt(setpath+'/ltY.txt')
        ltZ = np.loadtxt(setpath+'/ltZ.txt')
        rtX = np.loadtxt(setpath+'/rtX.txt')
        rtY = np.loadtxt(setpath+'/rtY.txt')
        rtZ = np.loadtxt(setpath+'/rtZ.txt')


        #biases removal from F/T
        lfX += blf[0]
        lfY += blf[1]
        lfZ += blf[2]
        rfX += brf[0]
        rfY += brf[1]
        rfZ += brf[2]
        ltX += blt[0]
        ltY += blt[1]
        ltZ += blt[2]
        rtX += brt[0]
        rtY += brt[1]
        rtZ += brt[2]
        self.blf = blf
        self.brf = brf
        self.blt = blt
        self.brt = brt
        if(self.gt_comparison):
            gt_lfZ  = np.loadtxt(setpath+'/gt_lfZ.txt')
            gt_rfZ  = np.loadtxt(setpath+'/gt_rfZ.txt')
            gt_lfX  = np.loadtxt(setpath+'/gt_lfX.txt')
            gt_rfX  = np.loadtxt(setpath+'/gt_rfX.txt')
            gt_lfY  = np.loadtxt(setpath+'/gt_lfY.txt')
            gt_rfY  = np.loadtxt(setpath+'/gt_rfY.txt')
            mu  = np.loadtxt(setpath+'/mu.txt')
       	    self.mu = mu
        if(self.comp_filtering):
            aX = np.loadtxt(setpath+'/gX.txt')
            aY = np.loadtxt(setpath+'/gY.txt')
            accX = np.loadtxt(setpath+'/accX.txt')
            accY = np.loadtxt(setpath+'/accY.txt')
            accZ = np.loadtxt(setpath+'/accZ.txt')
        else:
            roll_ = np.loadtxt(setpath+'/roll.txt')
            pitch_ = np.loadtxt(setpath + '/pitch.txt')



        dlen0 = np.size(cX)
        dlen1 = np.size(cY)
        dlen2 = np.size(lfZ)
        dlen6 = np.size(rfZ)
        if(self.comp_filtering):
            dlen3 = np.size(accZ)
        else:
            dlen3 = np.size(roll_)

        if(self.gt_comparison):
            dlen4 = np.size(gt_lfZ)
            dlen5 = np.size(gt_rfX)
            dlen = min(dlen0, dlen1, dlen2, dlen3, dlen4, dlen5,dlen6)
        else:
            dlen = min(dlen0, dlen1, dlen2, dlen3,dlen6)





        if(self.comp_filtering):
            roll = np.zeros((dlen))
            pitch = np.zeros((dlen))


        dcX = np.zeros((dlen))
        dcY = np.zeros((dlen))
        dcZ= np.zeros((dlen))
        droll = np.zeros((dlen))
        dpitch = np.zeros((dlen))

        if(self.gt_comparison):
            phase = np.zeros((dlen))




        for i in range(dlen):

            if(self.comp_filtering):
                roll[i], pitch[i] = self.compf.update(accX[i],accY[i],accZ[i],aX[i],aY[i])


            dcX[i]=self.cXdt.diff(cX[i])
            dcY[i]=self.cYdt.diff(cY[i])
            dcZ[i]=self.cZdt.diff(cZ[i])
            droll[i]=self.rolldt.diff(roll[i])
            dpitch[i]=self.pitchdt.diff(pitch[i])



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




        #Normalization + Standarization
        self.data_train = np.zeros((dlen, 11))
        self.data_train[:, 0] = lfX[0:dlen] - rfX[0:dlen]
        self.data_train[:, 1] = lfY[0:dlen] - rfY[0:dlen]
        self.data_train[:, 2] = lfZ[0:dlen] - rfZ[0:dlen]
        self.data_train[:, 3] = ltX[0:dlen] - rtX[0:dlen]
        self.data_train[:, 4] = ltY[0:dlen] - rtY[0:dlen]
        self.data_train[:, 5] = ltZ[0:dlen] - rtZ[0:dlen]


        self.data_train[1:dlen, 6] = dcX[1:dlen]
        self.data_train[1:dlen, 7] = dcY[1:dlen]
        self.data_train[1:dlen, 8] = dcZ[1:dlen]
        self.data_train[1:dlen, 9] = droll[1:dlen]
        self.data_train[1:dlen, 10] = dpitch[1:dlen]
        self.data_train[0, 6] = dcX[1]
        self.data_train[0, 7] = dcY[1]
        self.data_train[0, 8] = dcZ[1]
        self.data_train[0, 9] = droll[1]
        self.data_train[0, 10] = dpitch[1]








        self.dfX_train_min = min(self.data_train[:, 0])
        self.dfX_train_max = max(self.data_train[:, 0])
        self.dfX_train_mean = np.mean(self.data_train[:, 0])
        self.dfX_train_std = np.std(self.data_train[:, 0])


        self.dfY_train_min = min(self.data_train[:, 1])
        self.dfY_train_max = max(self.data_train[:, 1])
        self.dfY_train_mean = np.mean(self.data_train[:, 1])
        self.dfY_train_std = np.std(self.data_train[:, 1])

        self.dfZ_train_min = min(self.data_train[:, 2])
        self.dfZ_train_max = max(self.data_train[:, 2])
        self.dfZ_train_mean = np.mean(self.data_train[:, 2])
        self.dfZ_train_std = np.std(self.data_train[:, 2])

        self.dtX_train_min = min(self.data_train[:, 3])
        self.dtX_train_max = max(self.data_train[:, 3])
        self.dtX_train_mean = np.mean(self.data_train[:, 3])
        self.dtX_train_std = np.std(self.data_train[:, 3])

        self.dtY_train_min = min(self.data_train[:, 4])
        self.dtY_train_max = max(self.data_train[:, 4])
        self.dtY_train_mean = np.mean(self.data_train[:, 4])
        self.dtY_train_std = np.std(self.data_train[:, 4])

        self.dtZ_train_min = min(self.data_train[:, 5])
        self.dtZ_train_max = max(self.data_train[:, 5])
        self.dtZ_train_mean = np.mean(self.data_train[:, 5])
        self.dtZ_train_std = np.std(self.data_train[:, 5])


        self.dcX_train_min = min(self.data_train[:, 6])
        self.dcX_train_max = max(self.data_train[:, 6])
        self.dcX_train_mean = np.mean(self.data_train[:, 6])
        self.dcX_train_std = np.std(self.data_train[:, 6])


        self.dcY_train_min = min(self.data_train[:, 7])
        self.dcY_train_max = max(self.data_train[:, 7])
        self.dcY_train_mean = np.mean(self.data_train[:, 7])
        self.dcY_train_std = np.std(self.data_train[:, 7])


        self.dcZ_train_min = min(self.data_train[:, 8])
        self.dcZ_train_max = max(self.data_train[:, 8])
        self.dcZ_train_mean = np.mean(self.data_train[:, 8])
        self.dcZ_train_std = np.std(self.data_train[:, 8])


        self.droll_train_min = min(self.data_train[:, 9])
        self.droll_train_max = max(self.data_train[:, 9])
        self.droll_train_mean = np.mean(self.data_train[:, 9])
        self.droll_train_std = np.std(self.data_train[:, 9])

        self.dpitch_train_min = min(self.data_train[:, 10])
        self.dpitch_train_max = max(self.data_train[:, 10])
        self.dpitch_train_mean = np.mean(self.data_train[:, 10])
        self.dpitch_train_std = np.std(self.data_train[:, 10])



        '''
        self.data_train[:, 0] = self.normalize_data(self.data_train[:, 0],self.dfX_train_max, self.dfX_train_min)   
        self.data_train[:, 1] = self.normalize_data(self.data_train[:, 1],self.dfY_train_max, self.dfY_train_min)   
        self.data_train[:, 2] = self.normalize_data(self.data_train[:, 2],self.dfZ_train_max, self.dfZ_train_min)   
        self.data_train[:, 3] = self.normalize_data(self.data_train[:, 3],self.dtX_train_max, self.dtX_train_min)   
        self.data_train[:, 4] = self.normalize_data(self.data_train[:, 4],self.dtY_train_max, self.dtY_train_min)   
        self.data_train[:, 5] = self.normalize_data(self.data_train[:, 5],self.dtZ_train_max, self.dtZ_train_min)   
        self.data_train[:, 6] = self.normalize_data(self.data_train[:, 6],self.dcX_train_max, self.dcX_train_min)   
        self.data_train[:, 7] = self.normalize_data(self.data_train[:, 7],self.dcY_train_max, self.dcY_train_min)   
        self.data_train[:, 8] = self.normalize_data(self.data_train[:, 8],self.dcZ_train_max, self.dcZ_train_min)   
        self.data_train[:, 9] = self.normalize_data(self.data_train[:, 9],self.droll_train_max, self.droll_train_min)   
        self.data_train[:, 10] = self.normalize_data(self.data_train[:, 10],self.dpitch_train_max, self.dpitch_train_min)   
        
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
      


        if (self.gt_comparison):
            phase2=np.append([phase],[np.zeros_like(np.arange(cX.shape[0]-phase.shape[0]))])
            self.cX = cX[~(phase2==-1)]
            self.cY = cY[~(phase2==-1)]
            self.cZ = cZ[~(phase2==-1)]

            if self.comp_filtering:
                phase3=np.append([phase],[np.zeros_like(np.arange(accX.shape[0]-phase.shape[0]))])
                self.accX = accX[~(phase3==-1)]
                self.accY = accY[~(phase3==-1)]
                self.accZ = accZ[~(phase3==-1)]
                phase4=np.append([phase],[np.zeros_like(np.arange(aX.shape[0]-phase.shape[0]))])
                self.gX = aX[~(phase4==-1)]
                self.gY = aY[~(phase4==-1)]
            else:
                phase0=np.append([phase],[np.zeros_like(np.arange(roll.shape[0]-phase.shape[0]))])
                self.roll = roll[~(phase0==-1)]
                self.pitch = pitch[~(phase0==-1)]

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
        self.rolldt.reset()
        self.pitchdt.reset()
        if self.comp_filtering:
            self.compf.reset()





    def genInput(self,cX,cY,cZ,roll,pitch,lfX,lfY,lfZ,rfX,rfY,rfZ,ltX,ltY,ltZ,rtX,rtY,rtZ ,gt=None):

        if gt is None:
            gt=self

        output_ = np.zeros(11)
        dcX = gt.cXdt.diff(cX)
        dcY = gt.cYdt.diff(cY)
        dcZ = gt.cZdt.diff(cZ)
        droll = gt.rolldt.diff(roll)
        dpitch = gt.pitchdt.diff(pitch)

        lfX += gt.blf[0]
        lfY += gt.blf[1]
        lfZ += gt.blf[2]
        rfX += gt.brf[0]
        rfY += gt.brf[1]
        rfZ += gt.brf[2]
        ltX += gt.blt[0]
        ltY += gt.blt[1]
        ltZ += gt.blt[2]
        rtX += gt.brt[0]
        rtY += gt.brt[1]
        rtZ += gt.brt[2]

        dfZ = lfZ - rfZ
        dfX = lfX - rfX
        dfY = lfY - rfY
        dtZ = ltZ - rtZ
        dtX = ltX - rtX
        dtY = ltY - rtY


        if(dfX>gt.dfX_train_max):
            dfX=gt.dfX_train_max
        elif(dfX<gt.dfX_train_min):
            dfX=gt.dfX_train_min

        if(gt.dfX_train_max - gt.dfX_train_min != 0):
            dfX = (dfX - gt.dfX_train_min) / (gt.dfX_train_max - gt.dfX_train_min)
        else:
            dfX = 0.0




        if(dfY>gt.dfY_train_max):
            dfY=gt.dfY_train_max
        elif(dfY<gt.dfY_train_min):
            dfY=gt.dfY_train_min

        if(gt.dfY_train_max - gt.dfY_train_min != 0):
            dfY = (dfY - gt.dfY_train_min) / (gt.dfY_train_max - gt.dfY_train_min)
        else:
            dfY = 0.0

        if(dfZ>gt.dfZ_train_max):
            dfY=gt.dfZ_train_max
        elif(dfZ<gt.dfZ_train_min):
            dfZ=gt.dfZ_train_min

        if(gt.dfZ_train_max - gt.dfZ_train_min):
            dfZ = (dfZ - gt.dfZ_train_min) / (gt.dfZ_train_max - gt.dfZ_train_min)
        else:
            dfZ = 0.0

        if(dtX>gt.dtX_train_max):
            dtX=gt.dtX_train_max
        elif(dtX<gt.dtX_train_min):
            dtX=gt.dtX_train_min

        if(gt.dtX_train_max - gt.dtX_train_min):
            dtX = (dtX - gt.dtX_train_min) / (gt.dtX_train_max - gt.dtX_train_min)
        else:
            dtX = 0.0


        if(dtY>gt.dtY_train_max):
            dtY=gt.dtY_train_max
        elif(dtY<gt.dtY_train_min):
            dtY=gt.dtY_train_min

        if(gt.dtY_train_max - gt.dtY_train_min):
            dtY = (dtY - gt.dtY_train_min) / (gt.dtY_train_max - gt.dtY_train_min)
        else:
            dtY = 0.0


        if(dtZ>gt.dtZ_train_max):
            dtZ=gt.dtZ_train_max
        elif(dtZ<gt.dtZ_train_min):
            dtZ=gt.dtZ_train_min

        if(gt.dtZ_train_max - gt.dtZ_train_min):
            dtZ = (dtZ - gt.dtZ_train_min) / (gt.dtZ_train_max - gt.dtZ_train_min)
        else:
            dtZ = 0.0




        if(dcX>gt.dcX_train_max):
            dcX=gt.dcX_train_max
        elif(dcX<gt.dcX_train_min):
            dcX=gt.dcX_train_min

        if(gt.dcX_train_max - gt.dcX_train_min):
            dcX = (dcX -gt.dcX_train_min) / (gt.dcX_train_max - gt.dcX_train_min)
        else:
            dcX = 0.0



        if(dcY>gt.dcY_train_max):
            dcY=gt.dcY_train_max
        elif(dcY<gt.dcY_train_min):
            dcY=gt.dcY_train_min

        if(gt.dcY_train_max - gt.dcY_train_min):
            dcY = (dcY - gt.dcY_train_min) / (gt.dcY_train_max - gt.dcY_train_min)
        else:
            dcY = 0.0

        if(dcZ>gt.dcZ_train_max):
            dcZ=gt.dcZ_train_max
        elif(dcZ<gt.dcZ_train_min):
            dcZ=gt.dcZ_train_min

        if(gt.dcZ_train_max - gt.dcZ_train_min):
            dcZ = (dcZ - gt.dcZ_train_min) / (gt.dcZ_train_max - gt.dcZ_train_min)
        else:
            dcZ = 0.0

        if(droll>gt.droll_train_max):
            droll=gt.droll_train_max
        elif(droll<gt.droll_train_min):
            droll=gt.droll_train_min

        if(gt.droll_train_max - gt.droll_train_min):
            droll = (droll - gt.droll_train_min) / (gt.droll_train_max - gt.droll_train_min)
        else:
            droll = 0.0


        if(dpitch>gt.dpitch_train_max):
            droll=gt.dpitch_train_max
        elif(dpitch<gt.dpitch_train_min):
            dpitch=gt.dpitch_train_min

        if(gt.dpitch_train_max - gt.dpitch_train_min):
            dpitch = (dpitch - gt.dpitch_train_min) / (gt.dpitch_train_max - gt.dpitch_train_min)
        else:
            dpitch = 0.0



        output_[0] = dfX
        output_[1] = dfY
        output_[2] = dfZ
        output_[3] = dtX
        output_[4] = dtY
        output_[5] = dtZ
        output_[6] = dcX
        output_[7] = dcY
        output_[8] = dcZ
        output_[9] = droll
        output_[10] = dpitch

        return output_


    def normalize_data(self,din, dmax, dmin):    
        if(dmax-dmin != 0):
            dout = (din - dmin)/(dmax-dmin)
        else:
            dout =  np.zeros((np.size(din)))

        return dout

    def standarize_data(self,din,dmean,dstd):
        print(dmean,dstd)
        if(dstd != 0):
            dout = (din - dmean)/dstd
        else:
            dout =  np.zeros((np.size(din)))

        return dout


    def genInputCF(self,cX,cY,cZ,accX,accY, accZ, gX, gY, lfX,lfY,lfZ,rfX,rfY,rfZ,ltX,ltY,ltZ,rtX,rtY,rtZ, gt=None):


        if gt is None:
            gt=self

        roll, pitch = gt.compf.update(accX, accY, accZ, gX, gY)

        output_ = np.zeros(11)
        dcX = gt.cXdt.diff(cX)
        dcY = gt.cYdt.diff(cY)
        dcZ = gt.cZdt.diff(cZ)
        droll = gt.rolldt.diff(roll)
        dpitch = gt.pitchdt.diff(pitch)

        lfX += gt.blf[0]
        lfY += gt.blf[1]
        lfZ += gt.blf[2]
        rfX += gt.brf[0]
        rfY += gt.brf[1]
        rfZ += gt.brf[2]
        ltX += gt.blt[0]
        ltY += gt.blt[1]
        ltZ += gt.blt[2]
        rtX += gt.brt[0]
        rtY += gt.brt[1]
        rtZ += gt.brt[2]

        dfZ = lfZ - rfZ
        dfX = lfX - rfX
        dfY = lfY - rfY
        dtZ = ltZ - rtZ
        dtX = ltX - rtX
        dtY = ltY - rtY




        if(dfX>gt.dfX_train_max):
            dfX=gt.dfX_train_max
        elif(dfX<gt.dfX_train_min):
            dfX=gt.dfX_train_min

        if(gt.dfX_train_max - gt.dfX_train_min != 0):
            dfX = (dfX - gt.dfX_train_min) / (gt.dfX_train_max - gt.dfX_train_min)
        else:
            dfX = 0.0




        if(dfY>gt.dfY_train_max):
            dfY=gt.dfY_train_max
        elif(dfY<gt.dfY_train_min):
            dfY=gt.dfY_train_min

        if(gt.dfY_train_max - gt.dfY_train_min != 0):
            dfY = (dfY - gt.dfY_train_min) / (gt.dfY_train_max - gt.dfY_train_min)
        else:
            dfY = 0.0

        if(dfZ>gt.dfZ_train_max):
            dfY=gt.dfZ_train_max
        elif(dfZ<gt.dfZ_train_min):
            dfZ=gt.dfZ_train_min

        if(gt.dfZ_train_max - gt.dfZ_train_min):
            dfZ = (dfZ - gt.dfZ_train_min) / (gt.dfZ_train_max - gt.dfZ_train_min)
        else:
            dfZ = 0.0

        if(dtX>gt.dtX_train_max):
            dtX=gt.dtX_train_max
        elif(dtX<gt.dtX_train_min):
            dtX=gt.dtX_train_min

        if(gt.dtX_train_max - gt.dtX_train_min):
            dtX = (dtX - gt.dtX_train_min) / (gt.dtX_train_max - gt.dtX_train_min)
        else:
            dtX = 0.0


        if(dtY>gt.dtY_train_max):
            dtY=gt.dtY_train_max
        elif(dtY<gt.dtY_train_min):
            dtY=gt.dtY_train_min

        if(gt.dtY_train_max - gt.dtY_train_min):
            dtY = (dtY - gt.dtY_train_min) / (gt.dtY_train_max - gt.dtY_train_min)
        else:
            dtY = 0.0


        if(dtZ>gt.dtZ_train_max):
            dtZ=gt.dtZ_train_max
        elif(dtZ<gt.dtZ_train_min):
            dtZ=gt.dtZ_train_min

        if(gt.dtZ_train_max - gt.dtZ_train_min):
            dtZ = (dtZ - gt.dtZ_train_min) / (gt.dtZ_train_max - gt.dtZ_train_min)
        else:
            dtZ = 0.0




        if(dcX>gt.dcX_train_max):
            dcX=gt.dcX_train_max
        elif(dcX<gt.dcX_train_min):
            dcX=gt.dcX_train_min

        if(gt.dcX_train_max - gt.dcX_train_min):
            dcX = (dcX -gt.dcX_train_min) / (gt.dcX_train_max - gt.dcX_train_min)
        else:
            dcX = 0.0



        if(dcY>gt.dcY_train_max):
            dcY=gt.dcY_train_max
        elif(dcY<gt.dcY_train_min):
            dcY=gt.dcY_train_min

        if(gt.dcY_train_max - gt.dcY_train_min):
            dcY = (dcY - gt.dcY_train_min) / (gt.dcY_train_max - gt.dcY_train_min)
        else:
            dcY = 0.0

        if(dcZ>gt.dcZ_train_max):
            dcZ=gt.dcZ_train_max
        elif(dcZ<gt.dcZ_train_min):
            dcZ=gt.dcZ_train_min

        if(gt.dcZ_train_max - gt.dcZ_train_min):
            dcZ = (dcZ - gt.dcZ_train_min) / (gt.dcZ_train_max - gt.dcZ_train_min)
        else:
            dcZ = 0.0

        if(droll>gt.droll_train_max):
            droll=gt.droll_train_max
        elif(droll<gt.droll_train_min):
            droll=gt.droll_train_min

        if(gt.droll_train_max - gt.droll_train_min):
            droll = (droll - gt.droll_train_min) / (gt.droll_train_max - gt.droll_train_min)
        else:
            droll = 0.0


        if(dpitch>gt.dpitch_train_max):
            droll=gt.dpitch_train_max
        elif(dpitch<gt.dpitch_train_min):
            dpitch=gt.dpitch_train_min

        if(gt.dpitch_train_max - gt.dpitch_train_min):
            dpitch = (dpitch - gt.dpitch_train_min) / (gt.dpitch_train_max - gt.dpitch_train_min)
        else:
            dpitch = 0.0




        output_[0] = dfX
        output_[1] = dfY
        output_[2] = dfZ
        output_[3] = dtX
        output_[4] = dtY
        output_[5] = dtZ
        output_[6] = dcX
        output_[7] = dcY
        output_[8] = dcZ
        output_[9] = droll
        output_[10] = dpitch

        return output_




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
