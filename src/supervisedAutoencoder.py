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

