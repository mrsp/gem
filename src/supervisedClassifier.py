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
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.utils import plot_model
import keras.backend as K
import numpy as np


def clf_loss(y_true, y_pred):
    loss  = K.square(y_true[:,6] - (y_pred[:,0]*y_true[:,0] + y_pred[:,1]*y_true[:,3]))
    loss += K.square(y_true[:,7] - (y_pred[:,0]*y_true[:,1] + y_pred[:,1]*y_true[:,4]))
    loss += K.square(y_true[:,8] - (y_pred[:,0]*y_true[:,2] + y_pred[:,1]*y_true[:,5]))
    return K.mean(loss,axis = -1)


class supervisedClassifier():
    def __init__(self):
        self.firstrun = True

    def setDimensions(self, input_dim_, latent_dim, intermediate_dim):
        self.model = Sequential()
        self.model.add(Dense(30, activation='relu', input_dim=input_dim_))
        self.model.add(Dense(15, activation='relu'))
        self.model.add(Dense(2, activation='relu'))
        self.model.add(Dense(latent_dim, activation='softmax'))

        # Compile the model
        self.model.compile(optimizer='adam', 
                    loss=clf_loss, 
)
        #self.model.summary()
        self.firstrun = False


    def fit(self, x_train, y_train, epochs, batch_size):
        self.model_log = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,  verbose=1, shuffle=True)

