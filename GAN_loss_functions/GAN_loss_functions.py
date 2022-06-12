import tensorflow as tf
from tensorflow import keras

class GanLossFunctions:
    def __init__(self) -> None:
        self.binary_crossentropy = keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing = 0.1)

    def minimax_discriminator_loss(self, discriminator_gen_outputs, discriminator_real_outputs):
        loss_on_real = self.binary_crossentropy(tf.ones_like(discriminator_real_outputs), discriminator_real_outputs)
        loss_on_generated = self.binary_crossentropy(tf.zeros_like(discriminator_gen_outputs), discriminator_gen_outputs)
        total_loss = loss_on_real + loss_on_generated
        return total_loss

    def minimax_generator_loss(self, discriminator_gen_outputs):
        return self.binary_crossentropy(tf.ones_like(discriminator_gen_outputs), discriminator_gen_outputs)
        
    def modified_discriminator_loss(self, discriminator_gen_outputs, discriminator_real_outputs):
        return self.minimax_discriminator_loss(discriminator_gen_outputs, discriminator_real_outputs)

    def modified_generator_loss(self, discriminator_gen_outputs):
        return -1 * self.binary_crossentropy(tf.zeros_like(discriminator_gen_outputs), discriminator_gen_outputs)

    def wasserstein_discriminator_loss(self, discriminator_gen_outputs, discriminator_real_outputs, gradient_penalty, LAMBDA=10):
        return tf.math.reduce_mean(discriminator_gen_outputs) - tf.math.reduce_mean(discriminator_real_outputs) + LAMBDA * gradient_penalty

    def wasserstein_generator_loss(self, discriminator_gen_outputs):
        return -1 * tf.math.reduce_mean(discriminator_gen_outputs)
    
    def wasserstein_gradient_penalty(self, discriminator_model, gen_outputs, real_inputs, epsilon):
        mixed_inputs = epsilon * real_inputs + (1 - epsilon) * gen_outputs
        print(mixed_inputs.shape)
        with tf.GradientTape() as gradient_penalty_tape:
            gradient_penalty_tape.watch(mixed_inputs)
            discriminator_mixed_output = discriminator_model(mixed_inputs, training=True)
            gradients = gradient_penalty_tape.gradient(discriminator_mixed_output, mixed_inputs)
            gradient_norm = tf.norm(gradients, axis = 1)
            gradient_penalty = tf.reduce_mean(tf.square(gradient_norm - 1))
        return gradient_penalty
