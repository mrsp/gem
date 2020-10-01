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

from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.utils import plot_model
import keras.backend as K
import numpy as np
import tensorflow as tf
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 

def mae(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true)) 

def clf_loss(y_true, y_pred):
    #x  = 1.0 * K.square(y_true[:,6] - (y_pred[:,0]*y_true[:,0] + y_pred[:,1]*y_true[:,3]))
    #y  = 1.0 * K.square(y_true[:,7] - (y_pred[:,0]*y_true[:,1] + y_pred[:,1]*y_true[:,4]))
    #z  = 1.0 * K.square(y_true[:,8] - (y_pred[:,0]*y_true[:,2] + y_pred[:,1]*y_true[:,5]))
    #loss = K.mean(K.sqrt(x + y + z + K.epsilon()))
    #x  = 1.0 * tf.math.cosh(y_true[:,6] - (y_pred[:,0]*y_true[:,0] + y_pred[:,1]*y_true[:,3]))
    #y  = 1.0 * tf.math.cosh(y_true[:,7] - (y_pred[:,0]*y_true[:,1] + y_pred[:,1]*y_true[:,4]))
    #z  = 1.0 * tf.math.cosh(y_true[:,8] - (y_pred[:,0]*y_true[:,2] + y_pred[:,1]*y_true[:,5]))
    #loss = K.sum(tf.math.log(x + y + z), axis = -1)

    x  = 0.4 * K.abs(y_true[:,6] - (y_pred[:,0]*y_true[:,0] + y_pred[:,1]*y_true[:,3]))
    y  = 0.4 * K.abs(y_true[:,7] - (y_pred[:,0]*y_true[:,1] + y_pred[:,1]*y_true[:,4]))
    z  = 0.2 * K.abs(y_true[:,8] - (y_pred[:,0]*y_true[:,2] + y_pred[:,1]*y_true[:,5]))
    #loss = K.sum(x + y + z, axis = -1)
    loss = K.mean(x + y + z)

    return loss





class supervisedAutoencoder():
    def __init__(self):
        self.firstrun = True


    def setDimReduction(self, input_dim, latent_dim, intermediate_dim, num_classes):
        sae_input = Input(shape=(input_dim,), name='input')
        # this model maps an input to its encoded representation
        encoded = Dense(intermediate_dim, activation='tanh', name='encode_1', use_bias=False)(sae_input)
        encoded = Dense(latent_dim, activation='tanh', name='encode_2', use_bias=False)(encoded)
        predicted = Dense(latent_dim, activation='sigmoid', name='class_output', use_bias=True)(encoded)
        self.encoder = Model(sae_input, encoded)
        # Reconstruction Decoder: Latent to input
        decoded = Dense(intermediate_dim, activation='tanh', name='decode_1', use_bias=False)(encoded)
        decoded = Dense(input_dim, activation='tanh', name='reconst_output', use_bias=False)(decoded)
        # Take input and give classification and reconstruction
        self.model = Model(inputs=[sae_input], outputs=[decoded, predicted])
        self.model.compile(optimizer='adam',
                           loss={'class_output': clf_loss,
                                 'reconst_output': rmse},
                           loss_weights={'class_output': 1.0,
                                         'reconst_output': 0.1})
        #self.model.summary()
        self.firstrun = False
    
    def fit(self, x_train, y_train, x_validation, y_validation, epochs, batch_size):
        self.model_log = self.model.fit(x_train, {'reconst_output': x_train, 'class_output': y_train}, validation_data = (x_validation,{'reconst_output': x_validation, 'class_output': y_validation}), epochs=epochs, batch_size=batch_size,  verbose=1, shuffle=True)

