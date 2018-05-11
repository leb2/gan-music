import tensorflow as tf
import cPickle
from util_rnn import Util, DataHandler
from random import randint
import random
import numpy as np
import os
import sys

TIMESTEPS = 1
STEP = 1
MIDI_START = 21
MIDI_END = 108
INPUT_SIZE = MIDI_END - MIDI_START + 2
BATCH_SIZE = 2500


class Learner:
    def __init__(self, data_file, name):
        print("loading data...")
        self.dataset = cPickle.load(file(data_file))
        self.run_name = name

        self.batch_size = 180
        self.num_steps = 24
        self.data = DataHandler.preprocess(self.dataset, self.batch_size)

        print("initializing...")
        self.rnn_size = 64
        self.rnn_size_d = 64

        self.num_rnn_layers = 2
        self.num_rnn_layers_d = 2

        self.noise_dimension = 32
        self.embed_size = 50
        self.fixed_sample_noise = np.random.normal(size=(self.batch_size, self.noise_dimension))

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
            self.input_gen = tf.placeholder(tf.float32, shape=[self.batch_size, INPUT_SIZE])
            self.g_state = tf.placeholder(tf.float32, [self.num_rnn_layers, self.batch_size, self.rnn_size])
            self.g_noise = tf.placeholder(tf.float32, [self.batch_size, self.noise_dimension])
            g_state = tuple([self.g_state[i] for i in range(self.num_rnn_layers)])
            self.generated = self.generator(self.input_gen, g_state, self.g_noise)

        print("initializing discriminator")
        with tf.variable_scope('D') as scope:
            self.embedding = tf.Variable(tf.random_normal([INPUT_SIZE, self.embed_size], stddev=0.01))
            self.real_data = tf.placeholder(tf.int32, shape=[self.batch_size, self.num_steps])
            self.d_state_real = tf.placeholder(tf.float32, [self.num_rnn_layers_d, self.batch_size, self.rnn_size_d])
            self.d_state_fake = tf.placeholder(tf.float32, [self.num_rnn_layers_d, self.batch_size, self.rnn_size_d])

            real_data_embed = tf.nn.embedding_lookup(self.embedding, self.real_data)
            generated_embed = tf.tensordot(self.generated, self.embedding, [[2], [0]])

            d_state_real = tuple([self.d_state_real[i] for i in range(self.num_rnn_layers_d)])
            d_state_fake = tuple([self.d_state_fake[i] for i in range(self.num_rnn_layers_d)])

            discriminator_real, self.d_final_state_real = self.discriminator(real_data_embed, d_state_real)
            scope.reuse_variables()
            discriminator_fake, self.d_final_state_fake = self.discriminator(generated_embed, d_state_fake,
                                                                             reuse=True)
        print("initializing training")
        all_vars = tf.trainable_variables()
        g_params = [var for var in all_vars if var.name.startswith('G/')]
        d_params = [var for var in all_vars if var.name.startswith('D/')]

        self.loss_ind = tf.placeholder(tf.int32, shape=[])
        self.d_loss = tf.reduce_mean(-tf.log(discriminator_real[:, :, :])
                                     - tf.log(1 - discriminator_fake[:, :, :]))
        self.g_loss = tf.reduce_mean(-tf.log(discriminator_fake[:, :, :]))
        self.op_d = self.optimizer(self.d_loss, d_params)
        self.op_g = self.optimizer(self.g_loss, g_params)

        self.session = tf.Session()
        self.summary_disc_loss = tf.placeholder(tf.float32, shape=[])
        self.summary_gen_loss = tf.placeholder(tf.float32, shape=[])

        tf.summary.scalar('disc_loss', self.summary_disc_loss)
        tf.summary.scalar('gen_loss', self.summary_gen_loss)

        self.merged_summaries = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter('./summaries/' + self.run_name, self.session.graph)
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        print("finished initializing")

    def save_summaries(self, disc_loss, gen_loss, step):
        summary = self.session.run(self.merged_summaries, feed_dict={
            self.summary_disc_loss: disc_loss,
            self.summary_gen_loss: gen_loss
        })
        self.summary_writer.add_summary(summary, step)

    def save_model(self, name):
        save_path = self.saver.save(self.session, name + ".ckpt")
        print("Model saved in file: %s" % save_path)

    def load_model(self, name):
        self.saver.restore(self.session, name + ".ckpt")
        print("Model restored from %s" % name + ".ckpt")

    def generator(self, rnn_input, initial_states, noise):
        """
        :param initial_states: A tuple of the initial state for each rnn
        :param rnn_input: A tensor with shape [batch_size, input_size]
        :param noise: A tensor with shape [batch_size, noise_dimension] concatenated to each input
        :return: A tensor with shape [batch_size, num_steps, output_size]
        """
        generator = tf.contrib.rnn.MultiRNNCell([
            tf.contrib.rnn.GRUCell(self.rnn_size) for _ in range(self.num_rnn_layers)
        ])
        output, state = generator(tf.concat([rnn_input, noise], axis=1), initial_states)

        # output = tf.contrib.layers.fully_connected(output, self.g_hid_layer_size,
        #                                            scope='hidden_layer', activation_fn=tf.tanh)
        output = tf.layers.dense(output, INPUT_SIZE, name='output_layer', activation=tf.nn.softmax)
        outputs = [output]

        for i in range(1, self.num_steps):
            output, state = generator(tf.concat([output, noise], axis=1), state)
            # output = tf.contrib.layers.fully_connected(output, self.g_hid_layer_size,
            #                                            reuse=True, scope='hidden_layer', activation_fn=tf.tanh)
            output = tf.layers.dense(output, INPUT_SIZE, reuse=True, name='output_layer', activation=tf.nn.softmax)
            outputs.append(output)
            self.gen_final_output = output

        self.g_final_state = state
        return tf.stack(outputs, axis=1)

    def discriminator(self, rnn_input, initial_state, reuse=False):
        """
        :param initial_state: The initial state of the rnn
        :param reuse: Set to True if calling for second time onwards
        :param rnn_input: A tensor with shape [batch_size, num_steps, input_size]
        :return: A tensor with shape [batch_size, num_steps, 1]
        """
        discriminator = tf.contrib.rnn.MultiRNNCell([
            tf.contrib.rnn.GRUCell(self.rnn_size_d) for _ in range(self.num_rnn_layers_d)
        ])
        output, final_state = tf.nn.dynamic_rnn(discriminator, rnn_input, initial_state=initial_state)

        # weights_hid = tf.get_variable("hid_layer_w", initializer=tf.contrib.layers.xavier_initializer(),
        #                               shape=[self.rnn_size_d, self.d_hid_layer_size], dtype=tf.float32)
        # bias_hid = tf.get_variable("hid_layer_b", initializer=tf.zeros_initializer, shape=[self.d_hid_layer_size])
        # output = tf.tensordot(output, weights_hid, [[2], [0]]) + bias_hid
        # output = tf.nn.tanh(output)

        weights_out = tf.get_variable("out_layer_w", initializer=tf.contrib.layers.xavier_initializer(),
                                      shape=[self.rnn_size_d, 1], dtype=tf.float32)
        bias_out = tf.get_variable("out_layer_b", initializer=tf.zeros_initializer, shape=[1])
        output = tf.tensordot(output, weights_out, [[2], [0]]) + bias_out
        output = tf.nn.sigmoid(output)

        return output, final_state

    def sample_generator(self, iteration):
        batches = np.zeros((self.batch_size, 0, INPUT_SIZE))

        previous_output = np.zeros((self.batch_size, INPUT_SIZE))
        state_input = np.zeros((self.num_rnn_layers, self.batch_size, self.rnn_size))
        noise_input = self.fixed_sample_noise

        for i in range(15):
            batch, previous_output, state_input = self.session.run([
                self.generated, self.gen_final_output, self.g_final_state
            ], feed_dict={
                self.input_gen: previous_output,
                self.g_state: state_input,
                self.g_noise: noise_input
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
        return tf.train.AdamOptimizer(0.0007, beta1=0.5).minimize(loss, var_list=var_list)

    def train(self):
        total_loss_d = 0
        total_loss_g = 0
        total_len = 0
        d_counter = 0
        g_counter = 0
        num_batches = int(np.size(self.data) / (self.num_steps * self.batch_size))
        num_epochs = 500

        g_noise = np.random.normal(size=(self.batch_size, self.noise_dimension))
        g_state = np.zeros((self.num_rnn_layers, self.batch_size, self.rnn_size))
        d_state_fake = np.zeros((self.num_rnn_layers_d, self.batch_size, self.rnn_size_d))
        d_state_real = np.zeros((self.num_rnn_layers_d, self.batch_size, self.rnn_size_d))

        input_gen = np.zeros((self.batch_size, INPUT_SIZE))
        loss_d = 1.0

        # max_num_batches = 5  # Max number of batches before resetting the state
        max_len = self.num_steps  # Max of the sequence to take the loss of

        for epoch in range(num_epochs):
            # max_num_batches += 1

            times_trained_d = 0

            self.sample_generator(epoch)
            for batch_num in range(num_batches):
                # if batch_num % 300 == 0:
                # max_len += 1

                real_data = self.data[:, batch_num * self.num_steps: (batch_num + 1) * self.num_steps]
                seq_len = randint(1, max_len)
                ind = min(seq_len - 1, self.num_steps - 1)

                # Go through n batches before resetting state
                # if batch_num % max_num_batches == 0:
                if True:  # random.random() < 1.0 / max_num_batches:
                    g_noise = np.random.normal(size=(self.batch_size, self.noise_dimension))
                    g_state = np.zeros((self.num_rnn_layers, self.batch_size, self.rnn_size))
                    d_state_fake = np.zeros((self.num_rnn_layers_d, self.batch_size, self.rnn_size_d))
                    d_state_real = np.zeros((self.num_rnn_layers_d, self.batch_size, self.rnn_size_d))

                feed_dict = {
                    self.input_gen: input_gen,
                    self.real_data: real_data,

                    self.g_state: g_state,
                    self.loss_ind: ind,
                    self.g_noise: g_noise,

                    self.d_state_fake: d_state_fake,
                    self.d_state_real: d_state_real
                }

                ops = [self.d_loss, self.g_loss, self.d_final_state_real, self.d_final_state_fake,
                       self.gen_final_output, self.g_final_state]

                # Train generator
                if batch_num % 1 == 0:
                    ops.append(self.op_g)

                # Train discriminator
                if loss_d > 1.0 and batch_num % 1 == 0:
                    ops.append(self.op_d)
                    times_trained_d += 1

                vals = self.session.run(ops, feed_dict=feed_dict)
                loss_d, loss_g, d_final_state_real, d_final_state_fake, gen_final_output, g_final_state = vals[:6]

                total_loss_d += loss_d
                total_loss_g += loss_g
                d_state_fake = d_final_state_fake
                d_state_real = d_final_state_real
                g_state = g_final_state
                input_gen = gen_final_output

                d_counter += 1
                g_counter += 1
                total_len += seq_len

                if batch_num % 50 == 0:
                    print("Before batch %d/%d\tDiscriminator Loss: %.3f \t Generator Loss: %.3f \t Trained D: %d" %
                          (batch_num, num_batches, float(total_loss_d) / d_counter,
                           float(total_loss_g) / g_counter, times_trained_d))
                    times_trained_d = 0
            self.save_model("saved-weights/weights-" + str(epoch))
            self.save_summaries(float(total_loss_d) / d_counter, float(total_loss_g) / g_counter, epoch)
            total_len = total_loss_d = total_loss_g = d_counter = g_counter = 0


if __name__ == '__main__':
    run_name = sys.argv[1]
    learner = Learner('./data/MuseData.pickle', run_name)
    learner.train()
