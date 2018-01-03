import tensorflow as tf
import cPickle
from util_rnn import Util, DataHandler
from random import randint
import random
import numpy as np
import os

TIMESTEPS = 1
STEP = 1
MIDI_START = 21
MIDI_END = 108
INPUT_SIZE = MIDI_END - MIDI_START + 2
BATCH_SIZE = 2500


class Learner:
    def __init__(self, data_file):
        print("loading data...")
        self.dataset = cPickle.load(file(data_file))

        self.batch_size = 256
        self.num_steps = 20
        self.data = DataHandler.preprocess(self.dataset, self.batch_size)

        print("initializing...")
        self.rnn_size = 32
        self.rnn_size_d = 64
        self.g_hid_layer_size = 16
        self.d_hid_layer_size = 16

        self.noise_dimension = INPUT_SIZE
        self.embed_size = 50
        self.sample_noise = np.random.normal(size=(2, self.batch_size, self.rnn_size))

        self.gen_final_output = None
        self.g_final_state = None
        self.d_final_state_fake = None
        self.d_final_state_real = None
        self.d_losses = []
        self.g_losses = []
        self.op_ds = []
        self.op_gs = []

        print("initializing generator")
        with tf.variable_scope('G'):
            self.noise = tf.placeholder(tf.float32, shape=[self.batch_size, self.noise_dimension])
            self.g_state = tf.placeholder(tf.float32, [2, self.batch_size, self.rnn_size])
            g_state_tuple = tf.contrib.rnn.LSTMStateTuple(self.g_state[0], self.g_state[1])

            self.generated = self.generator(self.noise, g_state_tuple)

        print("initializing discriminator")
        with tf.variable_scope('D') as scope:
            self.real_data = tf.placeholder(tf.int32, shape=[self.batch_size, self.num_steps])
            self.embedding = tf.Variable(tf.random_normal([INPUT_SIZE, self.embed_size], stddev=0.01))
            real_data_embed = tf.nn.embedding_lookup(self.embedding, self.real_data)

            self.d_state_real = tf.placeholder(tf.float32, [2, self.batch_size, self.rnn_size_d])
            d_state_tuple_real = tf.contrib.rnn.LSTMStateTuple(self.d_state_real[0], self.d_state_real[1])

            # Add noise to real data
            real_data_embed += tf.random_normal([self.batch_size, self.num_steps, self.embed_size], stddev=0.01)

            discriminator_real, self.d_final_state_real = self.discriminator(real_data_embed, d_state_tuple_real)
            scope.reuse_variables()

            self.d_state_fake = tf.placeholder(tf.float32, [2, self.batch_size, self.rnn_size_d])
            d_state_tuple_fake = tf.contrib.rnn.LSTMStateTuple(self.d_state_fake[0], self.d_state_fake[1])

            generated_embed = tf.tensordot(self.generated, self.embedding, [[2], [0]])

            # Add noise to fake data
            generated_embed += tf.random_normal([self.batch_size, self.num_steps, self.embed_size], stddev=0.01)
            discriminator_fake, self.d_final_state_fake = self.discriminator(generated_embed, d_state_tuple_fake,
                                                                             reuse=True)
        print("initializing training")
        all_vars = tf.trainable_variables()
        g_params = [var for var in all_vars if var.name.startswith('G/')]
        d_params = [var for var in all_vars if var.name.startswith('D/')]

        self.loss_ind = tf.placeholder(tf.int32, shape=[])
        self.d_loss = tf.reduce_mean(-tf.log(discriminator_real[:, self.loss_ind, :])
                                     - tf.log(1 - discriminator_fake[:, self.loss_ind, :]))
        self.g_loss = tf.reduce_mean(-tf.log(discriminator_fake[:, self.loss_ind, :]))
        self.op_d = self.optimizer(self.d_loss, d_params)
        self.op_g = self.optimizer(self.g_loss, g_params)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        print("finished initializing")

    def save_model(self, name):
        save_path = self.saver.save(self.session, name + ".ckpt")
        print("Model saved in file: %s" % save_path)

    def load_model(self, name):
        self.saver.restore(self.session, name + ".ckpt")
        print("Model restored from %s" % name + ".ckpt")

    def generator(self, rnn_input, initial_state):
        """
        :param initial_state: The initial state of the rnn
        :param rnn_input: A tensor with shape [batch_size, num_steps, input_size]
        :return: A tensor with shape [batch_size, num_steps, output_size]
        """
        generator = tf.contrib.rnn.BasicLSTMCell(self.rnn_size)
        output, state = generator(rnn_input, initial_state)

        output = tf.contrib.layers.fully_connected(output, self.g_hid_layer_size,
                                                   scope='hidden_layer', activation_fn=tf.tanh)
        output = tf.contrib.layers.fully_connected(output, INPUT_SIZE, scope='output_layer', activation_fn=None)
        output = tf.nn.softmax(output)
        outputs = [output]

        for i in range(1, self.num_steps):
            output, state = generator(output, state)
            output = tf.contrib.layers.fully_connected(output, self.g_hid_layer_size,
                                                       reuse=True, scope='hidden_layer', activation_fn=tf.tanh)
            output = tf.contrib.layers.fully_connected(output, INPUT_SIZE, reuse=True,
                                                       scope='output_layer', activation_fn=None)
            output = tf.nn.softmax(output)
            outputs.append(output)
            self.gen_final_output = output

        self.g_final_state = state
        return tf.stack(outputs, axis=1)

    def discriminator(self, rnn_input, initial_state, reuse=False):
        """
        :param initial_state: The initial state of the rnn
        :param reuse: Set to True if calling for second time onwards
        :param rnn_input: A tensor with shape [batch_size, num_steps, input_size]
        :return: A tensor with shape [batch_size, 1, 1]
        """
        discriminator = tf.contrib.rnn.BasicLSTMCell(self.rnn_size_d)
        output, final_state = tf.nn.dynamic_rnn(discriminator, rnn_input, initial_state=initial_state)

        weights_hid = tf.get_variable("hid_layer_w", initializer=tf.contrib.layers.xavier_initializer(),
                                      shape=[self.rnn_size_d, self.d_hid_layer_size], dtype=tf.float32)
        bias_hid = tf.get_variable("hid_layer_b", initializer=tf.zeros_initializer, shape=[self.d_hid_layer_size])
        output = tf.tensordot(output, weights_hid, [[2], [0]]) + bias_hid
        output = tf.nn.tanh(output)

        weights_out = tf.get_variable("out_layer_w", initializer=tf.contrib.layers.xavier_initializer(),
                                      shape=[self.d_hid_layer_size, 1], dtype=tf.float32)
        bias_out = tf.get_variable("out_layer_b", initializer=tf.zeros_initializer, shape=[1])
        output = tf.tensordot(output, weights_out, [[2], [0]]) + bias_out
        output = tf.nn.sigmoid(output)

        return output, final_state

    def sample_generator(self, iteration):
        batches = np.zeros((self.batch_size, 0, INPUT_SIZE))
        for i in range(15):
            rnn_input = np.zeros((self.batch_size, INPUT_SIZE))
            state_input = self.sample_noise

            batch, rnn_input, state_input = self.session.run([
                self.generated, self.gen_final_output, self.g_final_state
            ], feed_dict={
                self.noise: rnn_input,
                self.g_state: state_input
            })
            batches = np.append(batches, batch, axis=1)

        for index, run in enumerate(batches[:min(2, batches.shape[0])]):
            piano_roll = []
            curr_timestep = []

            for probs in run:
                note_index = Util.sample(probs)
                if Util.is_continue_symbol(note_index):
                    piano_roll.append(curr_timestep)
                    curr_timestep = []
                else:
                    curr_timestep.append(Util.index_to_midi(note_index))
            if len(curr_timestep) != 0:
                piano_roll.append(curr_timestep)

            prefix = './samples/' + str(index) + '/'
            if not os.path.exists(prefix):
                os.makedirs(prefix)

            filename = prefix + ("%05d" % (iteration,))
            with open(filename + '.txt', 'w') as f:
                for item in piano_roll:
                    f.write("%s\n" % str(item))
            print("Saving midi file " + filename)
            Util.piano_roll_to_midi(piano_roll, filename)

    @staticmethod
    def optimizer(loss, var_list):
        return tf.train.AdamOptimizer(0.0007).minimize(loss, var_list=var_list)

    def train(self):
        total_loss_d = 0
        total_loss_g = 0
        total_len = 0
        d_counter = 0
        g_counter = 0
        num_batches = int(np.size(self.data) / (self.num_steps * self.batch_size))
        num_epochs = 250

        g_state_input = np.random.normal(size=(2, self.batch_size, self.rnn_size))
        d_state_fake_input = d_state_real_input = np.zeros((2, self.batch_size, self.rnn_size_d))

        # g_state_input = d_state_fake_input = d_state_real_input = random_state

        zero_input = np.zeros((self.batch_size, self.noise_dimension))
        loss_d = loss_g = 1.0

        max_num_batches = 5  # Max number of batches before resetting the state
        max_len = self.num_steps  # Max of the sequence to take the loss of

        for epoch in range(num_epochs):
            # max_num_batches += 1

            self.sample_generator(epoch)
            for batch_num in range(num_batches):
                # if batch_num % 300 == 0:
                # max_len += 1

                real_data = self.data[:, batch_num * self.num_steps: (batch_num + 1) * self.num_steps]
                seq_len = randint(1, max_len)
                ind = min(seq_len - 1, self.num_steps - 1)

                # Go through n batches before resetting state
                # if batch_num % max_num_batches == 0:
                if random.random() < 1.0 / max_num_batches:
                    g_state_input = np.random.normal(size=(2, self.batch_size, self.rnn_size))
                    d_state_fake_input = d_state_real_input = np.zeros((2, self.batch_size, self.rnn_size_d))

                feed_dict = {
                    self.noise: zero_input,
                    self.real_data: real_data,

                    self.g_state: g_state_input,
                    self.loss_ind: ind,

                    self.d_state_fake: d_state_fake_input,
                    self.d_state_real: d_state_real_input
                }

                if batch_num % 1 == 0:  # Train generator
                    return_vals = self.session.run([
                        self.d_loss, self.g_loss, self.d_final_state_real, self.d_final_state_fake,
                        self.gen_final_output, self.g_final_state, self.op_g
                    ], feed_dict=feed_dict)

                    loss_d, loss_g, d_final_state_real, d_final_state_fake, \
                        gen_final_output, g_final_state, _g = return_vals

                if batch_num % 30 == 0:  # Train discriminator
                    return_vals = self.session.run([
                        self.d_loss, self.g_loss, self.d_final_state_real, self.d_final_state_fake,
                        self.gen_final_output, self.g_final_state, self.op_d
                    ], feed_dict=feed_dict)

                    loss_d, loss_g, d_final_state_real, d_final_state_fake, \
                        gen_final_output, g_final_state, _d, = return_vals

                total_loss_d += loss_d
                total_loss_g += loss_g
                d_state_fake_input = d_final_state_fake
                d_state_real_input = d_final_state_real
                g_state_input = g_final_state
                zero_input = gen_final_output

                d_counter += 1
                g_counter += 1
                total_len += seq_len

                if batch_num % 10 == 0:
                    print("Before batch %d/%d\tDiscriminator Loss: %.3f \t Generator Loss: %.3f \t Max len: %.2f" %
                          (batch_num, num_batches, float(total_loss_d) / d_counter,
                           float(total_loss_g) / g_counter, max_len))
                    total_len = total_loss_d = total_loss_g = d_counter = g_counter = 0
            self.save_model("saved-weights/weights-" + str(epoch))


if __name__ == '__main__':
    learner = Learner('./data/MuseData.pickle')
    learner.train()
