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
from keras.losses import binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
import numpy as np

def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

class autoencoder():
    def __init__(self):
        self.firstrun = True

    def setDimReduction(self, input_dim, latent_dim, intermediate_dim):
        input_= Input(shape=(input_dim,))
        # "encoded" is the encoded representation of the input
        encoded = Dense(intermediate_dim, activation='linear',name='encode_1')(input_)
        encoded = Dense(latent_dim, activation='linear',name='encode_2')(encoded)
        ## "decoded" is the lossy reconstruction of the input
        decoded = Dense(intermediate_dim, activation='linear',name='decode_1')(encoded)
        decoded = Dense(input_dim, activation='linear',name='reconst_output')(decoded)
        # this model maps an input to its reconstruction
        self.model = Model(inputs=[input_], outputs=[decoded])
        # this model maps an input to its encoded representation
        self.encoder = Model(inputs=[input_], outputs=[encoded])
        self.model.compile(optimizer='adam', loss={"reconst_output":"mse"})
        #self.model.summary()
        self.firstrun = False

    def fit(self, x_train, x_validation, epochs, batch_size):
        self.model_log = self.model.fit(x_train, x_train, validation_data=(x_validation, x_validation), epochs=epochs, batch_size=batch_size,  verbose=1, shuffle=True)

 
        
        
      