from keras.layers import Input, Dense
from keras.models import Model
from keras.losses import binary_crossentropy
from keras.utils import plot_model
import numpy as np

class autoencoder():
    def __init__(self):
        self.firstrun = True

    def setDimReduction(self, input_dim, latent_dim, intermediate_dim):
        input_= Input(shape=(input_dim,))
        # "encoded" is the encoded representation of the input
        encoded = Dense(intermediate_dim, activation='selu')(input_)
        encoded = Dense(latent_dim, activation='selu')(encoded)
        ## "decoded" is the lossy reconstruction of the input
        decoded = Dense(intermediate_dim, activation='selu')(encoded)
        decoded = Dense(input_dim, activation='sigmoid')(decoded)
        # this model maps an input to its reconstruction
        self.model = Model(input_, decoded)
        # this model maps an input to its encoded representation
        self.encoder = Model(input_, encoded)
        # create a placeholder for an encoded (2-dimensional) input
        encoded_input = Input(shape=(latent_dim,))
        # retrieve the last layer of the autoencoder model
        deco = self.model.layers[-2](encoded_input)
        deco = self.model.layers[-1](deco)
        # create the decoder model
        self.decoder = Model(encoded_input, deco)
        self.model.compile(optimizer='adam', loss='binary_crossentropy')
        self.model.summary()
        self.firstrun = False

    def fit(self, x_train, epochs, batch_size):
        self.model_log = self.model.fit(x_train, x_train, epochs=epochs, batch_size=batch_size,  verbose=1, shuffle=True)

 
        
        
      