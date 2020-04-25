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
import numpy as np

class supervisedAutoencoder():
    def __init__(self):
        self.firstrun = True

    def setDimReduction(self, input_dim, latent_dim, intermediate_dim, num_classes):
        sae_input = Input(shape=(input_dim,), name='input')
        # Encoder: input to Z
        encoded = Dense(input_dim, activation='relu',
                        name='encode_1')(sae_input)
        encoded = Dense(intermediate_dim, activation='relu', name='encode_2')(encoded)
        encoded = Dense(latent_dim, activation='relu', name='z')(encoded)
        # Classification: Z to class
        predicted = Dense(num_classes, activation='softmax',
                          name='class_output')(encoded)
        # Reconstruction Decoder: Z to input
        decoded = Dense(latent_dim, activation='relu',
                        name='decode_1')(encoded)
        decoded = Dense(intermediate_dim, activation='relu', name='decode_2')(decoded)
        decoded = Dense(input_dim, activation='sigmoid',
                        name='reconst_output')(decoded)
        # Take input and give classification and reconstruction
        self.model = Model(inputs=[sae_input], outputs=[decoded, predicted])
        self.model.compile(optimizer='adam',
                           loss={'class_output': 'categorical_crossentropy',
                                 'reconst_output': 'binary_crossentropy'},
                           loss_weights={'class_output': 0.1,
                                         'reconst_output': 1.0})
        self.model.summary()
        self.firstrun = False

    def fit(self, x_train, y_train, epochs, batch_size):
        self.model_log = self.model.fit(x_train, {
                                          'reconst_output': x_train, 'class_output': y_train}, epochs=epochs, batch_size=batch_size,  verbose=1, shuffle=True)

