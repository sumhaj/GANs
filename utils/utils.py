import tensorflow as tf
import math
import matplotlib.pyplot as plt
from utils.constants import LATENT_SPACE_DIM

def load_dataset(path='mnist.npz'):
    return tf.keras.datasets.mnist.load_data(path)

def normalize_and_reshape_image(image_tensor, target_shape=None):
    current_height, current_width = image_tensor.shape[1:3]
    if target_shape is None:
        target_shape = current_width
    new_width = target_shape
    new_height = int((current_height * current_width) / new_width)
    print(current_height, new_height)
    image_tensor = image_tensor.reshape(image_tensor.shape[0], new_height, new_width, 1).astype('float32')
    image_tensor = (image_tensor - 127.5) / 127.5

def sample_gaussian_noise(batch_size):
    return tf.random.normal([batch_size, LATENT_SPACE_DIM])

def sample_uniform_noise(batch_size):
    return tf.random.uniform([batch_size], minval=0, maxval=10, dtype=tf.dtypes.int32)

def plot_and_save_images(epoch, saved_model_path, gan_type):
    random_tensor = sample_gaussian_noise(16)
    generator_model = tf.keras.models.load_model(saved_model_path)
    if gan_type == 1:
        random_labels = sample_uniform_noise(16)
        random_labels = tf.one_hot(random_labels, 10, dtype=tf.dtypes.float32)
        print(random_labels.shape)
        generated_image = generator_model(random_tensor, random_labels, training=False)
    else:
        generated_image = generator_model(random_tensor, training=False)
    

    fig = plt.figure(figsize=(4, 4))
    for i in range(generated_image.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(generated_image[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

        plt.savefig('output_images/image_at_epoch_{:04d}.png'.format(epoch))
        plt.show()
