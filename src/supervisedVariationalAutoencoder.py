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

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.losses import binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
import numpy as np

class supervisedVariationalAutoencoder():
    def __init__(self):
        self.firstrun = True
    
    # reparameterization trick
    # instead of sampling from Q(z|X), sample epsilon = N(0,I)
    # z = z_mean + sqrt(var) * epsilon
    def sampling(self,args):
        """Reparameterization trick by sampling from an isotropic unit Gaussian.
        # Arguments
            args (tensor): mean and log of variance of Q(z|X)
        # Returns
            z (tensor): sampled latent vector
        """
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon


    def setDimReduction(self, input_dim, latent_dim, intermediate_dim, num_classes):
        # VAE model = encoder + decoder
        # build encoder model
        inputs = Input(shape=(input_dim,), name='encoder_input')
        x = Dense(intermediate_dim, activation='relu')(inputs)
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)
        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(self.sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
        # instantiate encoder model
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        encoder.summary()
        #plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)


        # build decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(intermediate_dim, activation='relu')(latent_inputs)
        outputs = Dense(input_dim, activation='sigmoid')(x)
        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')
        decoder.summary()
        #plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

        # Add a classifier
        clf_latent_inputs = Input(shape=(latent_dim,), name='z_sampling_clf')
        clf_outputs = Dense(num_classes, activation='softmax',
                            name='class_output')(clf_latent_inputs)
        clf_supervised = Model(clf_latent_inputs, clf_outputs, name='clf')
        clf_supervised.summary()


        # instantiate VAE model
        # New: Add another output
        outputs = [decoder(encoder(inputs)[2]), clf_supervised(encoder(inputs)[2])]
        self.model = Model(inputs, outputs, name='vae_mlp')
        self.model.summary()

        reconstruction_loss = binary_crossentropy(inputs, outputs[0])
        reconstruction_loss *= input_dim

        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean((reconstruction_loss + kl_loss) / 100.0)
        self.model.add_loss(vae_loss)

        # New: add the clf loss
        self.model.compile(optimizer='adam', loss={'clf': 'categorical_crossentropy'},loss_weights={'clf': 0.1})
        self.model.summary()
        #plot_model(self.model, to_file='supervised_vae.png', show_shapes=True)

    def fit(self,x_train,y_train,epochs,batch_size):
        # reconstruction_loss = binary_crossentropy(inputs, outputs)
        self.model_log = self.model.fit(x_train, {'clf': y_train}, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)
