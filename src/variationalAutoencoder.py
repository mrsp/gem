from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.losses import binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
import numpy as np

class variationalAutoencoder():
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


    def setDimReduction(self, input_dim, latent_dim, intermediate_dim):
        # VAE model = encoder + decoder
        # build encoder model
        inputs = Input(shape=(input_dim,), name='encoder_input')
        x = Dense(intermediate_dim, activation='selu')(inputs)
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)
        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(self.sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
        # instantiate encoder model
        self.encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        self.encoder.summary()
        #plot_model(self.encoder, to_file='vae_mlp_encoder.png', show_shapes=True)
        # build decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(intermediate_dim, activation='selu')(latent_inputs)
        outputs = Dense(input_dim, activation='sigmoid')(x)
        # instantiate decoder model
        self.decoder = Model(latent_inputs, outputs, name='decoder')
        self.decoder.summary()
        #plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)
        # instantiate VAE model
        outputs = self.decoder(self.encoder(inputs)[2])
        self.model = Model(inputs, outputs, name='vae_mlp')
        self.model.summary()
        #reconstruction_loss = mse(inputs, outputs)
        reconstruction_loss = binary_crossentropy(inputs, outputs)
        reconstruction_loss *= input_dim
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.model.add_loss(vae_loss)
        #xent_loss = binary_crossentropy(x, x_decoded_mean)
        #kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
        #return xent_loss + kl_loss
        self.model.compile(optimizer='adam')
        self.model.summary()
        #plot_model(self.model, to_file='vae.png', show_shapes=True)
        self.firstrun = False

    def fit(self,x_train,epochs,batch_size):
        self.model_log = self.model.fit(x_train, epochs=epochs, batch_size=batch_size,verbose=1, shuffle=True)
