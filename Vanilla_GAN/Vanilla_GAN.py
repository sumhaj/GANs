import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
from utils.constants import LATENT_SPACE_DIM, MNIST_IMAGE_SIZE
# from GAN_loss_functions.GAN_loss_functions import GanLossFunctions

class vanilla_block(layers.Layer):
    def __init__(self, output_dim, normalize = True, dropout = 0.0, activation = None):
        super(vanilla_block, self).__init__()
        self.dense_layer = layers.Dense(output_dim)
        self.normalize = None
        self.dropout = None
        if activation == None:
            self.activation_function = layers.Activation(activations.relu)
        else:
            self.activation_function = activation

        if normalize == True:
            self.normalize = layers.BatchNormalization()
        if dropout != 0.0:
            self.dropout = layers.Dropout(rate=dropout)

    def call(self, input_features):
        x = self.dense_layer(input_features)
        output_features = self.activation_function(x)
        if self.normalize:
            output_features = self.normalize(output_features)
        if self.dropout:
            output_features = self.dropout(output_features)
        return output_features

class vanilla_generator(keras.Model):
    def __init__(self, image_size = MNIST_IMAGE_SIZE, LATENT_DIM = LATENT_SPACE_DIM):
        super(vanilla_generator, self).__init__()
        neurons_per_layer = [LATENT_DIM, 256, 512, 1024, image_size[0] * image_size[1]]
        self.vanilla_layer_1 = vanilla_block(neurons_per_layer[1])
        self.vanilla_layer_2 = vanilla_block(neurons_per_layer[2])
        self.vanilla_layer_3 = vanilla_block(neurons_per_layer[3])
        self.vanilla_layer_4 = vanilla_block(neurons_per_layer[4], normalize = False, activation = activations.tanh)
        self.reshape_layer = layers.Reshape(image_size)

    def call(self, random_vector):
        x = self.vanilla_layer_1(random_vector)
        x = self.vanilla_layer_2(x)
        x = self.vanilla_layer_3(x)
        x = self.vanilla_layer_4(x)
        return self.reshape_layer(x)


class vanilla_discriminator(keras.Model):
    def __init__(self, image_size = MNIST_IMAGE_SIZE):
        super(vanilla_discriminator, self).__init__()
        neurons_per_layer = [image_size[0] * image_size[1], 512, 256, 1]
        self.flatten = layers.Flatten()
        self.vanilla_layer_1 = vanilla_block(neurons_per_layer[1], normalize = False, dropout = 0.3)
        self.vanilla_layer_2 = vanilla_block(neurons_per_layer[2], normalize = False, dropout = 0.3)
        self.vanilla_layer_3 = vanilla_block(neurons_per_layer[3], normalize = False, activation = activations.sigmoid)

    def call(self, image_matrix):
        x = self.flatten(image_matrix)
        x = self.vanilla_layer_1(x)
        x = self.vanilla_layer_2(x)
        return self.vanilla_layer_3(x)