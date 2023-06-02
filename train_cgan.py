import tensorflow as tf
from GAN_loss_functions.GAN_loss_functions import GanLossFunctions
from cGAN.cGAN import ConditionalDiscriminator, ConditionalGenerator
from utils.constants import *
from utils.utils import *
import math
import matplotlib.pyplot as plt


class ConditionalGAN:
    def __init__(self, BATCH_SIZE = 32, loss_function = 'minimax_loss'):
        super(ConditionalGAN, self).__init__()
        self.generator_loss_function, self.discriminator_loss_function = self.__loss_function(loss_function)
        self.generator_model = ConditionalGenerator() 
        self.discriminator_model = ConditionalDiscriminator()
        self.mnist_dataset = load_mnist_dataset()
        self.batch_size = BATCH_SIZE
        self.gen_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-04, beta_1 = 0.9, beta_2 = 0.999)
        self.dis_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-04, beta_1 = 0.9, beta_2 = 0.999)
        self.buffer_size = 60000
        self.random_noise = LATENT_SPACE_DIM
    
    def __loss_function(self, loss_function):
        if loss_function == 'minimax_loss':
            return GanLossFunctions().minimax_generator_loss, GanLossFunctions().minimax_discriminator_loss
        elif loss_function == 'modified_loss':
            return GanLossFunctions().modified_generator_loss, GanLossFunctions().modified_discriminator_loss
        elif loss_function == 'wasserstein_loss':
            return GanLossFunctions().wasserstein_generator_loss, GanLossFunctions().wasserstein_discriminator_loss
        else:
            raise ValueError("Invalid loss function")

    def __train_step(self, train_image, train_label):
        random_tensor = sample_gaussian_noise(train_image.shape[0])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
            train_one_hot_encoded_label = tf.one_hot(train_label, 10)
            pred_image = self.generator_model(random_tensor, train_one_hot_encoded_label, training=True)
            fake_output = self.discriminator_model(pred_image, training=True)
            real_output = self.discriminator_model(train_image, training=True)
            generator_loss = self.generator_loss_function(fake_output)
            try:
                discriminator_loss = self.discriminator_loss_function(fake_output, real_output)
            except:
                epsilon = tf.random.normal([train_image.shape[0], 1, 1, 1])
                gradient_penalty = GanLossFunctions().wasserstein_gradient_penalty(self.discriminator_model, pred_image, train_image, epsilon)
                discriminator_loss = self.discriminator_loss_function(fake_output, real_output, gradient_penalty)
            generator_gradients = gen_tape.gradient(generator_loss, self.generator_model.trainable_variables)
            discriminator_gradients = dis_tape.gradient(discriminator_loss, self.discriminator_model.trainable_variables)
            self.gen_optimizer.apply_gradients(zip(generator_gradients, self.generator_model.trainable_variables))
            self.dis_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator_model.trainable_variables))
            return generator_loss, discriminator_loss

    def train(self, EPOCHS = 50):
        (train_images, train_labels), (_, _) = self.mnist_dataset
        # print(train_labels) # (60000, 28, 28) 
        normalize_and_reshape_image(train_images)
        total_images = train_images.shape[0]
        # train_images = train_images.reshape(total_images, 28, 28, 1).astype('float32') # adding channel dim and changing datatype to float 32 bit
        # train_images = train_images - 127.5 / 127.5 # normalizing pixel values to range between -1 and 1
        total_batches = math.ceil(total_images / self.batch_size)
        train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(self.buffer_size).batch(self.batch_size)
        train_labels = tf.data.Dataset.from_tensor_slices(train_labels).shuffle(self.buffer_size).batch(self.batch_size)
        
        for epoch in range(EPOCHS):
            batch_number = 1
            print("EPOCH : {}".format(epoch+1))
            for train_image_batch, train_labels_batch in zip(train_dataset, train_labels):
                generator_loss, discriminator_loss = self.__train_step(train_image_batch, train_labels_batch)
                print("Epoch Number : {epoch} | Batch Number : {batch} of total_batches :{total_batches} | Generator Loss : {gen_loss} | Discriminator Loss : {dis_loss}".format(epoch=epoch+1,
                batch=batch_number, total_batches=total_batches, gen_loss=generator_loss, dis_loss=discriminator_loss))
                batch_number += 1
           
            self.generator_model.save("saved_gan_models/saved_epoch_{}_conditional_generator_model".format(epoch+1))
            self.discriminator_model.save("saved_gan_models/saved_epoch_{}_conditional_discriminator_model".format(epoch+1))
            plot_and_save_images(epoch+1, "saved_gan_models/saved_epoch_{}_conditional_generator_model".format(epoch+1), GANType().CGAN)
            
        self.generator_model.save("saved_gan_models/saved_conditional_generator_model")
        self.discriminator_model.save("saved_gan_models/saved_conditional_dsicriminator_model")