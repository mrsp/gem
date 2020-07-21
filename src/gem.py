#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
 * GeM - Gait-phase Estimation Module
 *
 * Copyright 2018-2020 Stylianos Piperakis, Foundation for Research and Technology Hellas (FORTH)
 * License: BSD
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
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


from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

from sklearn import mixture
from sklearn.cluster import KMeans
#import keras
from keras.layers import Input, Dense
from keras.models import Model
from Gaussian import Gaussian
from variationalAutoencoder import variationalAutoencoder
from autoencoder import autoencoder
from supervisedAutoencoder import supervisedAutoencoder
from supervisedVariationalAutoencoder import supervisedVariationalAutoencoder
class GeM():
    def __init__(self):
        self.gmm = mixture.GaussianMixture(n_components=3, covariance_type='full', max_iter=100, tol=7e-3, init_params = 'kmeans', n_init=30,warm_start=False,verbose=1)
        self.kmeans = KMeans(init='k-means++',n_clusters=3, n_init=500,tol=6.5e-2)
        self.pca_dim = False
        self.gmm_cl_id = False
        self.kmeans_cl_id = False
        self.gs = Gaussian()
        self.ef = 0.1
        self.firstrun = True

	
    def setDimReduction(self, dim_):
        self.latent_dim = dim_
        self.input_dim = 21
        self.intermidiate_dim = 10
        self.pca = PCA(n_components=self.latent_dim)
        self.ae = autoencoder()
        self.ae.setDimReduction(self.input_dim, self.latent_dim, self.intermidiate_dim)
        self.vae = variationalAutoencoder()
        self.vae.setDimReduction(self.input_dim, self.latent_dim, self.intermidiate_dim)
        self.sae = supervisedAutoencoder()
        self.sae.setDimReduction(self.input_dim, self.latent_dim, self.intermidiate_dim, 2)
        self.svae = supervisedVariationalAutoencoder()
        self.svae.setDimReduction(self.input_dim, self.latent_dim, self.intermidiate_dim, 2)

    def setFrames(self,lfoot_frame_,rfoot_frame_):
        self.lfoot_frame = lfoot_frame_
        self.rfoot_frame = rfoot_frame_

    def fit(self,data_train,red,cl,data_labels = None):
        self.red = red
        print("Data Size ",data_train.size)
        self.data_train = data_train
        if red == 'pca':
            print("Dimensionality reduction with PCA")
            self.reducePCA(data_train)
        elif red == 'autoencoders':
            print("Dimensionality reduction with autoencoders")
            self.reduceAE(data_train)
        elif red == "variationalAutoencoders":
            print("Dimensionality reduction with variational autoencoders")
            self.reduceVAE(data_train)
        elif red == "supervisedAutoencoders":
            print("Dimensionality reduction with supervised autoencoders")
            self.reduceSAE(data_train,data_labels)
        elif red == "supervisedVariationalAutoencoder":
            print("Dimensionality reduction with supervised variational autoencoders")
            self.reduceSVAE(data_train,data_labels)
        else:
            self.reduced_data_train = data_train
            print("Choose a valid dimensionality reduction method")

        
        if cl == 'gmm':
            print("Clustering with Gaussian Mixture Models")
            self.clusterGMM()
            self.gmm_cl_id = True
        elif cl == 'kmeans':
            print("Clustering with Kmeans")
            self.clusterKMeans()
            self.kmeans_cl_id = True
        else:
            print("Choose a valid clustering method")

        self.firstrun = False



    def predict(self, data_):
        if(self.red == 'pca'):
            reduced_data = self.pca.transform(data_.reshape(1,-1))
        elif(self.red == 'autoencoders'):
            reduced_data = self.ae.encoder.predict(data_.reshape(1,-1))
        elif(self.red == 'variationalAutoencoders'):
            reduced_data = self.vae.encoder.predict(data_.reshape(1,-1))[0]
        elif(self.red == 'supervisedAutoencoders'):
            reduced_data = self.sae.encoder.predict(data_.reshape(1,-1))
        elif(self.red == 'supervisedVariationalAutoencoders'):
            reduced_data = self.svae.encoder.predict(data_.reshape(1,-1))[0]
        else:
            print('Unrecognired Training Method')
            reduced_data = data_

        if(self.gmm_cl_id):
            gait_phase = self.gmm.predict(reduced_data), reduced_data
        elif(self.kmeans_cl_id):
            gait_phase = self.kmeans.predict(reduced_data), reduced_data
        else:
            print('Unrecognired Clustering Method')

        if(self.firstrun == False):
            if(gait_phase == 0):
                self.support_leg = self.lfoot_frame
            elif(gait_phase == 1):
                self.support_leg = self.rfoot_frame
        else:
            if(data_[2]>0):
                self.support_leg = self.lfoot_frame
            else:
                self.support_leg = self.rfoot_frame
            
            self.firstrun = False

        return gait_phase 

    def reducePCA(self,data_train):
        self.pca.fit(data_train)
        self.reduced_data_train = self.pca.transform(data_train)
        self.pca_dim = True
        print("Explained variance ratio")
        print(self.pca.explained_variance_ratio_)
        print("Reprojection Error")
        print(mean_squared_error(data_train, self.pca.inverse_transform(self.reduced_data_train)))

    def reduceAE(self,data_train):
        self.ae.fit(data_train, 20, 32)
        self.reduced_data_train =  self.ae.encoder.predict(data_train)
        self.pca_dim = False

    def reduceSAE(self,data_train,data_labels):
        self.sae.fit(data_train,data_labels, 20, 32)
        self.reduced_data_train =  self.sae.encoder.predict(data_train)
        self.pca_dim = False

    def reduceSVAE(self,data_train,data_labels):
        self.svae.fit(data_train,data_labels, 20, 32)
        self.reduced_data_train =  self.svae.encoder.predict(data_train)[0]
        self.pca_dim = False

    def reduceVAE(self,data_train):
        self.vae.fit(data_train,20,32)
        self.reduced_data_train =  self.vae.encoder.predict(data_train)[0]
        self.pca_dim = False

    def clusterGMM(self):
        self.gmm.fit(self.reduced_data_train)
        self.predicted_labels_train = self.gmm.predict(self.reduced_data_train)

    def clusterKMeans(self):
        self.kmeans.fit(self.reduced_data_train)
        self.predicted_labels_train = self.kmeans.predict(self.reduced_data_train)


    def getSupportLeg(self):
        return self.support_leg


    def getLLegProb(self):
        return self.pl

    def getRLegProb(self):
        return self.pr

    def computeForceContactProb(self,  fmin,  sigma,  f):
        return 1.000 - self.gs.cdf(fmin, f, sigma)

    def computeCOPContactProb(self, max,  min,  sigma,  cop):
        if (cop != 0):
            return self.gs.cdf(max, cop, sigma) - self.gs.cdf(min, cop, sigma)
        else:
            return 0

    def computeKinContactProb(self,  vmin,  sigma,  v):
        return 1.000 - self.gs.cdf(vmin, v, sigma)

    def computeForceProb(self,lf,  rf, lfmin, rfmin, sigmalf, sigmarf):
        plf = self.computeForceContactProb(lfmin, sigmalf, lf)
        prf = self.computeForceContactProb(rfmin, sigmarf, rf)
        self.pr = prf 
        self.pl = plf 
    
    def computeGRFProb(self,lf,rf,mass,g):
        lf = self.cropGRF(lf,mass,g)
        rf = self.cropGRF(rf,mass,g)
        p = lf + rf + 2.0 * self.ef
        plf = 0
        prf = 0
        if (p > 0):
            plf = (lf+self.ef) / p
            prf = (rf+self.ef) / p
        
        if(plf < 0):
            plf = 0
        
        if(plf > 1.0):
            plf = 1.0

        if(prf < 0):
            prf = 0

        if(prf > 1.0):
            prf = 1.0

        self.pl = plf
        self.pr = prf

    def cropGRF(self,f_, mass_, g_):
        return max(0.0, min(f_, mass_ * g_))

        
    def computeContactProb(self,  coplx,  coply,  coprx,  copry, xmax, xmin, ymax, ymin,  sigmalc, sigmarc):
        plc = self.computeCOPContactProb(xmax, xmin, sigmalc, coplx) * self.computeCOPContactProb(ymax, ymin, sigmalc, coply)
        prc = self.computeCOPContactProb(xmax, xmin, sigmarc, coprx) * self.computeCOPContactProb(ymax, ymin, sigmarc, copry)
        self.pr = self.pr * prc
        self.pl = self.pl * plc

    def computeVelProb(self, lv, rv, lvelTresh, rvelTresh, sigmalv, sigmarv):
        plv =  self.computeKinContactProb(lvelTresh,  sigmalv,  np.linalg.norm(lv))
        prv =  self.computeKinContactProb(rvelTresh,  sigmarv,  np.linalg.norm(rv))

        self.pr = self.pr * prv
        self.pl = self.pl * plv

    def predictFT(self, lf,  rf, lfmin, rfmin, sigmalf, sigmarf, useCOP=False, coplx=0,  coply=0,  coprx=0,  copry=0, xmax=0, xmin=0, ymax=0, ymin=0,  sigmalc=0, sigmarc=0):
        self.computeForceProb(lf,  rf, lfmin, rfmin, sigmalf, sigmarf)

        if(useCOP):
            self.computeContactProb(coplx,  coply,  coprx,  copry, xmax, xmin, ymax, ymin,  sigmalc, sigmarc)

        p = self.pl + self.pr

        if (p != 0):
            self.pl = self.pl / p 
            self.pr = self.pr / p 
        else:
            self.pl = 0
            self.pr = 0

        if(self.firstrun):
            if(self.pl > self.pr):
                self.support_leg = self.lfoot_frame
            else:
                self.support_leg = self.rfoot_frame
            self.firstrun = False



        if (self.pl > 0.35 and self.pr <= 0.35):
            gait_phase = 0
            self.support_leg = self.lfoot_frame
        elif(self.pr > 0.35 and self.pl <= 0.35):
            gait_phase = 1
            self.support_leg = self.rfoot_frame
        elif(self.pr > 0.35 and self.pl > 0.35):
            gait_phase = 2
        else:
            gait_phase = -1

        return gait_phase



    def predictGRF(self, lf,  rf, mass, g, useCOP=False, coplx=0,  coply=0,  coprx=0,  copry=0, xmax=0, xmin=0, ymax=0, ymin=0,  sigmalc=0, sigmarc=0):
        self.computeGRFProb(lf,  rf, mass, g)

        if(useCOP):
            self.computeContactProb(coplx,  coply,  coprx,  copry, xmax, xmin, ymax, ymin, sigmalc, sigmarc)

       
        p = self.pl + self.pr

        if (p != 0):
            self.pl = self.pl / p 
            self.pr = self.pr / p 
        else:
            self.pl = 0
            self.pr = 0
            
        if(self.firstrun):
            if(self.pl > self.pr):
                self.support_leg = self.lfoot_frame
            else:
                self.support_leg = self.rfoot_frame
            self.firstrun = False
        
 
        if (self.pl > 0.35 and self.pr <= 0.35):
            gait_phase = 0
            self.support_leg = self.lfoot_frame
        elif(self.pr > 0.35 and self.pl <= 0.35):
            gait_phase = 1
            self.support_leg = self.rfoot_frame
        elif(self.pr >= 0.35 and self.pl >= 0.35):
            gait_phase = 2
        else:
            gait_phase = -1

        return gait_phase
