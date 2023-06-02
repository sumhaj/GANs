import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
# from Vanilla_GAN.Vanilla_GAN import vanilla_block
from utils.constants import LATENT_SPACE_DIM, MNIST_IMAGE_SIZE, MNIST_NUMBER_CLASSES

class dcgan_upsample_block(layers.Layer):
    def __init__(self, channels, normalize=True, dropout=0.0, activation=None):
        self.transpose_conv_layer = layers.Conv2DTranspose(channels)
        if normalize:
            output_features = layers.BatchNormalization()
        
        if activation == None:
            self.activation_function = layers.Activation(activations.relu)
        else:
            self.activation_function = activation

        if dropout != 0.0:
            self.dropout = layers.Dropout(rate=dropout)
    
    def call(self, input_features):
        output_features = self.transpose_conv_layer(input_features)
        if self.normalize:
            output_features = self.normalize(output_features)
        output_features = self.activation_function(x=output_features)
        if self.dropout:
            output_features = self.dropout(output_features)
        return output_features
        
        

class dcgan_generator(keras.Model):
    def __init__(self, image_dim=[218, 178, 3]):
        super(dcgan_generator, self).__init__()
        self.dense_layer = layers.Dense(1000, activation='relu')
        self.reshape_layer = layers.Reshape(())
        self.upsample_block_1 = dcgan_upsample_block(32)
        self.upsample_block_2 = dcgan_upsample_block(64)
        self.upsample_block_3 = dcgan_upsample_block(128, activation=activations.tanh)

    def call(self, random_vector):
        x = self.dense_layer(random_vector)
        x = self.reshape_layer(x)
        x = self.upsample_block_1(x)
        x = self.upsample_block_2(x)
        return  self.upsample_block_3(x)
    

class dcgan_downsample_block(layers.Layer):
    def __init__(self, channels, normalize=True, dropout=0.0, activation=None):
        self.conv_layer = layers.Conv2D(channels)
        if normalize:
            output_features = layers.BatchNormalization()
        
        if activation == None:
            self.activation_function = layers.Activation(activations.relu)
        else:
            self.activation_function = activation

        if dropout != 0.0:
            self.dropout = layers.Dropout(rate=dropout)
    
    def call(self, input_features):
        output_features = self.conv_layer(input_features)
        if self.normalize:
            output_features = self.normalize(output_features)
        output_features = self.activation_function(x=output_features)
        if self.dropout:
            output_features = self.dropout(output_features)
        return output_features


class dcgan_discriminator(keras.Model):
    def __init__(self):
        super(dcgan_discriminator, self).__init__()
        self.downsample_block_1 = dcgan_downsample_block(128, normalize=False, dropout=0.2)
        self.downsample_block_2 = dcgan_downsample_block(64, normalize=False, dropout=0.2)
        self.downsample_block_3 = dcgan_downsample_block(32, normalize=False, dropout=0.2)
        self.dense_layer_1 = layers.Dense(100, activation='relu')
        self.dense_layer_2 = layers.Dense(1, activation='sigmoid')

    def call(self, img_tensor):
        x = self.downsample_block_1(img_tensor)
        x = self.downsample_block_2(x)
        x = self.downsample_block_3(x)
        x = self.dense_layer_1(x)
        return self.dense_layer_2(x)


