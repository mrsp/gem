#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
 * GeM - Gait-phase Estimation Module
 *
 * Copyright 2017-2018 Stylianos Piperakis and Stavros Timotheatos, Foundation for Research and Technology Hellas (FORTH)
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
import keras
from keras.layers import Input, Dense
from keras.models import Model


class GeM():

    def __init__(self):
        self.pca = PCA(n_components=2)
        self.gmm = mixture.BayesianGaussianMixture(weight_concentration_prior_type = "dirichlet_process",n_components=3, covariance_type='full', max_iter=10000, tol=1e-6, init_params = 'kmeans', n_init=10, random_state=0)
        self.kmeans = KMeans(init='k-means++',n_clusters=3, n_init=100)
        self.pca_dim = False
        self.gmm_cl_id = False
        self.kmeans_cl_id = False



        input_= Input(shape=(6,))
        # "encoded" is the encoded representation of the input
        encoded = Dense(4, activation='linear')(input_)
        encoded = Dense(2, activation='linear')(encoded)
        ## "decoded" is the lossy reconstruction of the input
        decoded = Dense(4, activation='linear')(encoded)
        decoded = Dense(6, activation='linear')(decoded)
        # this model maps an input to its reconstruction
        self.autoencoder = Model(input_, decoded)
        # this model maps an input to its encoded representation
        self.encoder = Model(input_, encoded)
        # create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=(2,))
        # retrieve the last layer of the autoencoder model
        deco = self.autoencoder.layers[-2](encoded_input)
        # deco = autoencoder.layers[-2](deco)
        deco = self.autoencoder.layers[-1](deco)
        # create the decoder model
        self.decoder = Model(encoded_input, deco)
        self.autoencoder.compile(optimizer='adam', loss='mean_squared_error')


    def fit(self,data_train,red,cl):
        if red == 'pca':
            print("Dimensionality reduction with PCA")
            self.reducePCA(data_train)
        elif red == 'autoencoders':
            print("Dimensionality reduction with autoencoders")
            self.reduceAE(data_train)


        else:
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
            print('Uncomment')


        if(self.gmm_cl_id):
            return self.gmm.predict(reduced_data), reduced_data
        elif(self.kmeans_cl_id):
            return self.kmeans.predict(reduced_data), reduced_data
        else:
            print('Error')

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
                             epochs=5,
                             batch_size=14,
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






