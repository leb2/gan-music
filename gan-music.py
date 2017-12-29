import tensorflow as tf
import numpy as np
import os
from scipy import misc
import sys
from util import Util

class Gan:
    def __init__(self, name):
        self.batch_size = 50
        self.noise_dimension = 16
        self.embedding_size = 16

        # Data shape [num_timesteps, 88]
        self.data = np.load('./data/midi.npy')  # Shape [398560, 88]
        self.data_dim = (32, 88)  # orig (1200, 1600)
        self.sample_noise = np.random.normal(size=(self.batch_size, self.noise_dimension))
        self.run_name = name
        self.save_real_samples()

        self.embedding = tf.Variable(tf.random_normal(shape=(Util.INPUT_SIZE, self.embedding_size), stddev=0.001))

        with tf.variable_scope('G'):
            self.noise = tf.placeholder(tf.float32, shape=[self.batch_size, self.noise_dimension])
            self.generated = self.generator(self.noise)

        with tf.variable_scope('D') as scope:
            self.real_data = tf.placeholder(tf.float32,
                    shape=[self.batch_size, self.data_dim[0], self.data_dim[1], 1], name='real_data')

            real_data = self.real_data # + tf.random_normal(self.real_data.shape, stddev=0.1)
            discriminator_real = self.discriminator(real_data)
            scope.reuse_variables()

            fake_data = self.generated  # + tf.random_normal(self.generated.shape, stddev=0.1)
            discriminator_fake = self.discriminator(fake_data, reuse=True)

        # To prevent numerical instabilities
        disc_real = tf.maximum(discriminator_real, 1e-9)
        disc_fake = tf.maximum(discriminator_fake, 1e-9)

        self.loss_d = tf.reduce_mean(-tf.log(disc_real) - tf.log(1 - disc_fake))
        self.loss_g = tf.reduce_mean(-tf.log(disc_fake))

        all_vars = tf.trainable_variables()
        g_params = [var for var in all_vars if var.name.startswith('G/')]
        d_params = [var for var in all_vars if var.name.startswith('D/')]

        self.opt_d = self.optimizer(self.loss_d, d_params)
        self.opt_g = self.optimizer(self.loss_g, g_params)

        self.session = tf.Session()

        self.summary_disc_loss = tf.placeholder(tf.float32, shape=[])
        self.summary_gen_loss = tf.placeholder(tf.float32, shape=[])
        self.summary_images = tf.placeholder(tf.float32, shape=[self.batch_size, self.data_dim[0], self.data_dim[1], 1])
        tf.summary.image('sample_images', self.summary_images)

        tf.summary.scalar('disc_loss', self.summary_disc_loss)
        tf.summary.scalar('gen_loss', self.summary_gen_loss)

        self.merged_summaries = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter('./summaries/' + self.run_name, self.session.graph)
        self.session.run(tf.global_variables_initializer())

    def save_real_samples(self):
        print("Saving real samples...")
        prefix = './real/'
        Util.tensor_to_midi('real/real1', self.data[5000:5500])
        image = np.repeat(np.expand_dims(self.data[50:50+32], axis=2), 3, axis=2)
        misc.imsave(prefix + 'real2' + '.jpg', image)
        print("Done saving real samples")

    def save_summaries(self, disc_loss, gen_loss, images, step):
        summary = self.session.run(self.merged_summaries, feed_dict={
            self.summary_disc_loss: disc_loss,
            self.summary_gen_loss: gen_loss,
            self.summary_images: images
        })
        self.summary_writer.add_summary(summary, step)

    def generator(self, noise_input, reuse=False):
        # output = tf.layers.dense(noise_input, 256, activation=tf.nn.tanh)
        output = tf.layers.dense(noise_input, 4 * 9 * 8, activation=tf.nn.tanh)
        output = tf.reshape(output, shape=[self.batch_size, 4, 9, 8])
        output = self.conv_transpose(output, name='transpose1', filters=128, strides=(2, 3), kernel_size=5)
        output = self.conv_transpose(output, name='transpose3', filters=64, strides=2, kernel_size=5)
        output = self.conv_transpose(output, name='transpose4', filters=64, strides=2, kernel_size=5)

        print("Generator pre resize shape:")
        print(output.shape)

        output = tf.image.resize_images(output, self.data_dim)
        output = self.conv_transpose(output, name='transpose6', filters=64, activation=tf.nn.relu, padding='same')
        output = self.conv_transpose(output, name='transpose7', filters=64, activation=tf.nn.relu, padding='same')
        output = self.conv_transpose(output, name='transpose8', filters=1, activation=tf.sigmoid, padding='same')
        return output

    @staticmethod
    def conv_transpose(tensor, filters=16, kernel_size=(1, 3), name=None, activation=tf.nn.relu, strides=1,
                       padding='same'):
        return tf.layers.conv2d_transpose(tensor, filters, kernel_size, strides=strides,
                padding=padding, name=name, activation=activation)

    @staticmethod
    def discriminator(d_input, reuse=False, filters=8):
        """
        :param d_input: Tensor with dimension [batch_size, num_timesteps, 88]
        :param reuse: True for the second time onwards calling this method
        :param filters: Number of filters in convolution layers
        :return: Tensor dimension [batch_size, 1]
        """
        padding = 'same'

        # Embedding is size [88, embedding_size]
        # output = tf.tensordot(self.embedding, d_input, [[0], [2]])
        # output = tf.transpose(output, perm=[1, 2, 0])
        output = tf.layers.conv2d(d_input, filters=filters, kernel_size=3, padding=padding, reuse=reuse, name='conv1',
                                  activation=tf.nn.leaky_relu)
        output = tf.contrib.layers.max_pool2d(output, kernel_size=3, stride=3, padding=padding)
        output = tf.layers.conv2d(output, filters=3, kernel_size=3, padding=padding, reuse=reuse, name='conv2',
                                  activation=tf.nn.leaky_relu)
        output = tf.layers.conv2d(output, filters=3, kernel_size=3, padding=padding, reuse=reuse, name='conv3',
                                  activation=tf.nn.leaky_relu)
        output = tf.contrib.layers.max_pool2d(output, kernel_size=3, stride=3, padding=padding)
        output = tf.contrib.layers.flatten(output)

        output = tf.layers.dense(output, 256, activation=tf.nn.tanh, reuse=reuse, name='fc1')
        output = tf.layers.dense(output, 1, activation=tf.sigmoid, reuse=reuse, name='fc2')

        return output

    @staticmethod
    def optimizer(loss, var_list):
        return tf.train.AdamOptimizer(0.0005, beta1=0.5) \
            .minimize(loss, var_list=var_list)

    def sample_generator(self, midi_name, num_samples=3):
        generated = self.session.run([self.generated], feed_dict={self.noise: self.sample_noise})[0]
        for i in range(1, num_samples + 1):
            prefix = './samples/' + self.run_name + '/' + str(i) + '/'
            if not os.path.exists(prefix):
                os.makedirs(prefix)
            Util.tensor_to_midi(prefix + midi_name, np.squeeze(generated)[i])
        # print("Saved %d samples" % num_samples)
        return generated

    def train(self, num_iterations=1000):
        total_loss_d = total_loss_g = 0
        d_counter = g_counter = 0
        avg_disc_loss = avg_gen_loss = 0
        prev_loss_d = 1.0

        # Original data [num_timesteps, 88]
        sequence_length = self.data_dim[0]

        # New data [num_sequences, sequence_length, 88]

        self.sample_generator("%05d" % (0,))

        for i in range(1, num_iterations + 1):
            print("Epoch %d" % i)
            data = self.data[i % 32:]
            num_sequences = int(data.shape[0] / sequence_length)
            data = data[:sequence_length * num_sequences]
            data = np.reshape(data, [num_sequences, self.data_dim[0], self.data_dim[1], 1])
            num_batches = int(num_sequences / self.batch_size)

            for batch_num in range(num_batches):
                noise = np.random.normal(size=(self.batch_size, self.noise_dimension))
                real_data = data[batch_num * self.batch_size: (batch_num + 1) * self.batch_size]

                feed_dict = {
                    self.noise: noise,
                    self.real_data: real_data
                }

                # Update discriminator
                if batch_num % 1 == 0:
                    ops = [self.loss_d]
                    if prev_loss_d > 0.8:  # Only train if loss is greater than threshold
                        ops.append(self.opt_d)

                    prev_loss_d = self.session.run(ops, feed_dict=feed_dict)[0]
                    total_loss_d += prev_loss_d
                    d_counter += 1

                # Update generator
                if batch_num % 1 == 0:
                    total_loss_g += self.session.run([self.loss_g, self.opt_g], feed_dict)[0]
                    g_counter += 1

                if (batch_num + 1) % 50 == 0 or (batch_num + 1) == num_batches:
                    avg_disc_loss = float(total_loss_d) / d_counter if d_counter > 0 else 0
                    avg_gen_loss = float(total_loss_g) / g_counter if d_counter > 0 else 0
                    print("Before batch %d/%d\tDiscriminator Loss: %.3f \t Generator Loss: %.3f" %
                          (batch_num + 1, num_batches, avg_disc_loss, avg_gen_loss))

            images = self.sample_generator("%05d" % (i,))
            self.save_summaries(avg_disc_loss, avg_gen_loss, images, step=i)
            total_loss_d = total_loss_g = g_counter = d_counter = 0
            print("")


if __name__ == "__main__":
    name = sys.argv[1]
    gan = Gan(name)
    gan.train(500)




