#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
 * GeM - Gait-phase Estimation Module
 *
 * Copyright 2019-2020 Stylianos Piperakis, Foundation for Research and Technology Hellas (FORTH)
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

class GeM():
    def __init__(self):
        self.gmm = mixture.GaussianMixture(n_components=3, covariance_type='full', max_iter=100, tol=7e-3, init_params = 'kmeans', n_init=30,warm_start=False,verbose=1)
        self.kmeans = KMeans(init='k-means++',n_clusters=3, n_init=500,tol=6.5e-2)
        self.pca_dim = False
        self.gmm_cl_id = False
        self.kmeans_cl_id = False
        self.gs = Gaussian()

        self.firstrun = True

	
    def setDimReduction(self, dim_):
        self.red_dim = dim_
        self.pca = PCA(n_components=self.red_dim)
        input_= Input(shape=(11,))
        
        if(dim_ == 2):
            # "encoded" is the encoded representation of the input
            encoded = Dense(5, activation='selu')(input_)
            encoded = Dense(2, activation='selu')(encoded)
            ## "decoded" is the lossy reconstruction of the input
            decoded = Dense(5, activation='selu')(encoded)
            decoded = Dense(11, activation='selu')(decoded)
            # this model maps an input to its reconstruction
            self.autoencoder = Model(input_, decoded)
            # this model maps an input to its encoded representation
            self.encoder = Model(input_, encoded)
            # create a placeholder for an encoded (32-dimensional) input
            encoded_input = Input(shape=(2,))
            # retrieve the last layer of the autoencoder model
            deco = self.autoencoder.layers[-2](encoded_input)
            #deco = self.autoencoder.layers[-2](deco)
            deco = self.autoencoder.layers[-1](deco)
            # create the decoder model
            self.decoder = Model(encoded_input, deco)
            self.autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        elif(dim_ == 3):
              # "encoded" is the encoded representation of the input
            encoded = Dense(6, activation='selu')(input_)
            encoded = Dense(3, activation='selu')(encoded)
            ## "decoded" is the lossy reconstruction of the input
            decoded = Dense(6, activation='selu')(encoded)
            decoded = Dense(11, activation='selu')(decoded)
            # this model maps an input to its reconstruction
            self.autoencoder = Model(input_, decoded)
            # this model maps an input to its encoded representation
            self.encoder = Model(input_, encoded)
            # create a placeholder for an encoded (32-dimensional) input
            encoded_input = Input(shape=(3,))
            # retrieve the last layer of the autoencoder model
            deco = self.autoencoder.layers[-2](encoded_input)
            #deco = self.autoencoder.layers[-2](deco)
            deco = self.autoencoder.layers[-1](deco)
            # create the decoder model
            self.decoder = Model(encoded_input, deco)
            self.autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        elif(dim_ == 4):
              # "encoded" is the encoded representation of the input
            encoded = Dense(8, activation='selu')(input_)
            encoded = Dense(4, activation='selu')(encoded)
            ## "decoded" is the lossy reconstruction of the input
            decoded = Dense(8, activation='selu')(encoded)
            decoded = Dense(11, activation='selu')(decoded)
            # this model maps an input to its reconstruction
            self.autoencoder = Model(input_, decoded)
            # this model maps an input to its encoded representation
            self.encoder = Model(input_, encoded)
            # create a placeholder for an encoded (32-dimensional) input
            encoded_input = Input(shape=(4,))
            # retrieve the last layer of the autoencoder model
            deco = self.autoencoder.layers[-2](encoded_input)
            #deco = self.autoencoder.layers[-2](deco)
            deco = self.autoencoder.layers[-1](deco)
            # create the decoder model
            self.decoder = Model(encoded_input, deco)
            self.autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        elif(dim_ == 5):
              # "encoded" is the encoded representation of the input
            encoded = Dense(8, activation='selu')(input_)
            encoded = Dense(5, activation='selu')(encoded)
            ## "decoded" is the lossy reconstruction of the input
            decoded = Dense(8, activation='selu')(encoded)
            decoded = Dense(11, activation='selu')(decoded)
            # this model maps an input to its reconstruction
            self.autoencoder = Model(input_, decoded)
            # this model maps an input to its encoded representation
            self.encoder = Model(input_, encoded)
            # create a placeholder for an encoded (32-dimensional) input
            encoded_input = Input(shape=(5,))
            # retrieve the last layer of the autoencoder model
            deco = self.autoencoder.layers[-2](encoded_input)
            #deco = self.autoencoder.layers[-2](deco)
            deco = self.autoencoder.layers[-1](deco)
            # create the decoder model
            self.decoder = Model(encoded_input, deco)
            self.autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        elif(dim_ == 6):
              # "encoded" is the encoded representation of the input
            encoded = Dense(6, activation='selu')(input_)
            ## "decoded" is the lossy reconstruction of the input
            decoded = Dense(11, activation='selu')(encoded)
            # this model maps an input to its reconstruction
            self.autoencoder = Model(input_, decoded)
            # this model maps an input to its encoded representation
            self.encoder = Model(input_, encoded)
            # create a placeholder for an encoded (32-dimensional) input
            encoded_input = Input(shape=(6,))
            # retrieve the last layer of the autoencoder model
            #deco = self.autoencoder.layers[-2](encoded_input)
            #deco = self.autoencoder.layers[-2](deco)
            deco = self.autoencoder.layers[-1](encoded_input)
            # create the decoder model
            self.decoder = Model(encoded_input, deco)
            self.autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    def setFrames(self,lfoot_frame_,rfoot_frame_):
        self.lfoot_frame = lfoot_frame_
        self.rfoot_frame = rfoot_frame_

    def fit(self,data_train,red,cl):
        print("Data Size ",data_train.size)
        self.data_train = data_train
        if red == 'pca':
            print("Dimensionality reduction with PCA")
            self.reducePCA(data_train)
        elif red == 'autoencoders':
            print("Dimensionality reduction with autoencoders")
            self.reduceAE(data_train)


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



    def predict(self, data_):


        if(self.pca_dim):
            reduced_data = self.pca.transform(data_.reshape(1,-1))
        else:
            reduced_data = self.encoder.predict(data_.reshape(1,-1))


        if(self.gmm_cl_id):
            gait_phase = self.gmm.predict(reduced_data), reduced_data
        elif(self.kmeans_cl_id):
            gait_phase = self.kmeans.predict(reduced_data), reduced_data
        else:
            print('Unrecognired Training Module')

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
        
        self.autoencoder.fit(data_train, data_train,
                             epochs=100,
                             batch_size=7,
                             shuffle=True,
                             validation_data=(data_train, data_train))
        self.reduced_data_train =  self.encoder.predict(data_train)
        
        self.pca_dim = False

    def clusterGMM(self):
        self.gmm.fit(self.reduced_data_train)
        self.predicted_labels_train = self.gmm.predict(self.reduced_data_train)

    def clusterKMeans(self):
        self.kmeans.fit(self.reduced_data_train)
        self.predicted_labels_train = self.kmeans.predict(self.reduced_data_train)


    def getSupportLeg(self):
        return self.support_leg

    def computeForceContactProb(self,  fmin,  sigma,  f):
        return 1.000 - self.gs.cdf(fmin, f, sigma)

    def computeCOPContactProb(self, max,  min,  sigma,  cop):
        if (cop != 0):
            return self.gs.cdf(max, cop, sigma) - self.gs.cdf(min, cop, sigma)
        else:
            return 0

    def computeKinContactProb(self,  vmin,  sigma,  v):
        return 1.000 - self.gs.cdf(vmin, v, sigma)

    def computeForceProb(self,lf,  rf, sigmalf, sigmarf):
        plf = self.computeForceContactProb(lfmin, sigmalf, lf)
        prf = self.computeForceContactProb(rfmin, sigmarf, rf)
        self.pr = prf 
        self.pl = plf 

    def computeContactProb(self,  coplx,  coply,  coprx,  copry, xmax, xmin, ymax, ymin, lfmin, rfmin, sigmalc, sigmarc):

       
        plc = self.computeCOPContactProb(xmax, xmin, sigmalc, coplx) * self.computeCOPContactProb(ymax, ymin, sigmalc, coply)
        prc = self.computeCOPContactProb(xmax, xmin, sigmarc, coprx) * self.computeCOPContactProb(ymax, ymin, sigmarc, copry)
        self.pr = self.pr * prc
        self.pl = self.pl * plc

    def computeVelProb(self, lv, rv, lvelTresh, rvelTresh, sigmalv, sigmarv):
        plv =  self.computeKinContactProb(lvelTresh,  sigmalv,  np.linalg.norm(lv))
        prv =  self.computeKinContactProb(rvelTresh,  sigmarv,  np.linalg.norm(rv))

        self.pr = self.pr * prv
        self.pl = self.pl * plv

    def predictFT(self, lf,  rf, lfmin, rfmin, sigmalf, sigmarf, useCOP, coplx,  coply,  coprx,  copry, xmax, xmin, ymax, ymin,  sigmalc, sigmarc):

        self.computeForceProb(lf,  rf, sigmalf, sigmarf)

        if(useCOP):
            self.computeContactProb(coplx,  coply,  coprx,  copry, xmax, xmin, ymax, ymin, lfmin, rfmin, sigmalc, sigmarc)

        p = self.pl + self.pr

        if (p != 0):
            self.pl = self.pl / p + 0.5
            self.pr = self.pr / p + 0.5
        else:
            self.pl = 0
            self.pr = 0

        if(self.firstrun):
            if(self.pl > self.pr):
                self.support_leg = self.lfoot_frame
            else:
                self.support_leg = self.rfoot_frame
            self.firstrun = False



        if (self.pl > 0.5 and self.pr <= 0.5):
            gait_phase = 0
            self.support_leg = self.lfoot_frame
        elif(self.pr > 0.5 and self.pl <= 0.5):
            gait_phase = 1
            self.support_leg = self.rfoot_frame
        elif(self.pr > 0.5 and self.pl > 0.5):
            gait_phase = 2
        else:
            gait_phase = -1

        return gait_phase



    def predictFTKin(self, lf,  rf, lfmin, rfmin, sigmalf, sigmarf, useCOP, coplx,  coply,  coprx,  copry, xmax, xmin, ymax, ymin,  sigmalc, sigmarc, useKin, lv, rv, lvelTresh, rvelTresh, sigmalv, sigmarv):
        self.computeForceProb(lf,  rf, sigmalf, sigmarf)

        if(useCOP):
            self.computeContactProb(coplx,  coply,  coprx,  copry, xmax, xmin, ymax, ymin, lfmin, rfmin, sigmalc, sigmarc)

        if(useKin):
            self.computeVelProb(lv, rv, lvelTresh, rvelTresh, sigmalv, sigmarv)

        p = self.pl + self.pr

        if (p != 0):
            self.pl = self.pl / p + 0.5
            self.pr = self.pr / p + 0.5
        else:
            self.pl = 0
            self.pr = 0
            
        if(self.firstrun):
            if(self.pl > self.pr):
                self.support_leg = self.lfoot_frame
            else:
                self.support_leg = self.rfoot_frame
            self.firstrun = False
        
 
        if (self.pl > 0.5 and self.pr <= 0.5):
            gait_phase = 0
            self.support_leg = self.lfoot_frame
        elif(self.pr > 0.5 and self.pl <= 0.5):
            gait_phase = 1
            self.support_leg = self.rfoot_frame
        elif(self.pr > 0.5 and self.pl > 0.5):
            gait_phase = 2
        else:
            gait_phase = -1

        return gait_phase
