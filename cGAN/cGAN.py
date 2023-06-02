import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
from Vanilla_GAN.Vanilla_GAN import vanilla_block
from utils.constants import LATENT_SPACE_DIM, MNIST_IMAGE_SIZE, MNIST_NUMBER_CLASSES

class ConditionalGenerator(keras.Model):
    def __init__(self, image_size = MNIST_IMAGE_SIZE, LATENT_DIM = LATENT_SPACE_DIM, number_of_classes = MNIST_NUMBER_CLASSES):
        super(ConditionalGenerator, self).__init__()
        neurons_per_layer = [LATENT_DIM + number_of_classes, 256, 512, image_size[0]*image_size[1]*image_size[2]]
        self.concat_layer = layers.Concatenate(axis=-1)
        self.vanilla_layer_1 = vanilla_block(neurons_per_layer[1])
        self.vanilla_layer_2 = vanilla_block(neurons_per_layer[2])
        self.vanilla_layer_3 = vanilla_block(neurons_per_layer[3], normalize=False, activation=activations.tanh)
        self.reshape_layer = layers.Reshape(image_size)

    def call(self, input_latent_space, input_class_one_hot_encoding):
        x = self.concat_layer([input_latent_space, input_class_one_hot_encoding])
        x = self.vanilla_layer_1(x)
        x = self.vanilla_layer_2(x)
        x = self.vanilla_layer_3(x)
        return self.reshape_layer(x)
        
    
class ConditionalDiscriminator(keras.Model):
    def __init__(self, image_size = MNIST_IMAGE_SIZE):
        super(ConditionalDiscriminator, self).__init__()
        neurons_per_layer = [image_size[0] * image_size[1] * image_size[2], 512, 256, 10]
        self.flatten_layer = layers.Flatten()
        self.vanilla_layer_1 = vanilla_block(neurons_per_layer[1], normalize=False, dropout=0.3)
        self.vanilla_layer_2 = vanilla_block(neurons_per_layer[2], normalize=False, dropout=0.3)
        self.vanilla_layer_3 = vanilla_block(neurons_per_layer[3], normalize=False, dropout=0.3, activation=activations.softmax)

    def call(self, input_image):
        x = self.flatten_layer(input_image)
        x = self.vanilla_layer_1(x)
        x = self.vanilla_layer_2(x)
        return self.vanilla_layer_3(x)


        


                                        
