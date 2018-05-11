from __future__ import division
import tensorflow as tf
import cPickle
from util_rnn import Util, DataHandler
import numpy as np
import os
import sys

MIDI_START = 21
MIDI_END = 108
INPUT_SIZE = MIDI_END - MIDI_START + 2


class Learner:
    def __init__(self, data_file, name):
        print("Loading data...")
        self.dataset = cPickle.load(file(data_file))
        self.run_name = name

        self.batch_size = 256
        self.num_steps = 24
        self.data = DataHandler.preprocess(self.dataset, self.batch_size)

        print("Initializing...")
        self.rnn_size = 128
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
        self.g_final_state_single = None
        self.gen_final_output_single = None
        self.d_losses = []
        self.g_losses = []
        self.op_ds = []
        self.op_gs = []

        print("Initializing generator")
        with tf.variable_scope('G') as scope:
            self.generator_cell = tf.contrib.rnn.MultiRNNCell([
                tf.contrib.rnn.GRUCell(self.rnn_size, reuse=tf.get_variable_scope().reuse)
                for _ in range(self.num_rnn_layers)
            ])

            # -- GENERATION -- #
            self.input_gen = tf.placeholder(tf.int32, shape=[self.batch_size, self.num_steps], name="input_multiple")
            self.g_state = tf.placeholder(tf.float32, [self.num_rnn_layers, self.batch_size, self.rnn_size],
                                          name="state")
            self.g_noise = tf.placeholder(tf.float32, [self.batch_size, self.noise_dimension], name="noise")
            self.g_index = tf.placeholder(tf.int32, [])

            g_state = tuple([self.g_state[i] for i in range(self.num_rnn_layers)])
            input_gen = tf.one_hot(self.input_gen, depth=INPUT_SIZE, axis=-1)

            # Shape [batch_size, num_steps, INPUT_SIZE]
            self.generated = self.generator(input_gen, g_state, self.g_noise, self.g_index)
            scope.reuse_variables()

            self.input_gen_single = tf.placeholder(tf.int32, shape=[self.batch_size], name='input_single')
            input_gen_single = tf.one_hot(self.input_gen_single, depth=INPUT_SIZE, axis=-1)
            self.generated_single = self.generator_single(input_gen_single, g_state, self.g_noise)

        print("Initializing discriminator")
        self.embedding = tf.Variable(tf.random_normal([INPUT_SIZE, self.embed_size], stddev=0.01))
        self.real_data = tf.placeholder(tf.int32, shape=[self.batch_size, self.num_steps], name="real_data")
        self.fake_data = tf.placeholder(tf.int32, shape=[self.batch_size, self.num_steps], name="fake_data")

        with tf.variable_scope('D') as scope:
            self.d_state_real = tf.placeholder(tf.float32, [self.num_rnn_layers_d, self.batch_size, self.rnn_size_d],
                                               name="state_real")
            self.d_state_fake = tf.placeholder(tf.float32, [self.num_rnn_layers_d, self.batch_size, self.rnn_size_d],
                                               name="state_fake")

            self.discriminator_cell = tf.contrib.rnn.MultiRNNCell([
                tf.contrib.rnn.GRUCell(self.rnn_size_d) for _ in range(self.num_rnn_layers_d)
            ])
            d_state_real = tuple([self.d_state_real[i] for i in range(self.num_rnn_layers_d)])
            d_state_fake = tuple([self.d_state_fake[i] for i in range(self.num_rnn_layers_d)])

            real_data_embed = tf.nn.embedding_lookup(self.embedding, self.real_data)
            generated_embed = tf.nn.embedding_lookup(self.embedding, self.fake_data)

            self.discriminator_real, self.d_final_state_real = self.discriminator(real_data_embed, d_state_real)
            scope.reuse_variables()
            self.discriminator_fake, self.d_final_state_fake = self.discriminator(generated_embed, d_state_fake)

        print("initializing training")
        all_vars = tf.trainable_variables()
        g_params = [var for var in all_vars if var.name.startswith('G/')]
        d_params = [var for var in all_vars if var.name.startswith('D/')]

        self.loss_ind = tf.placeholder(tf.int32, shape=[])
        self.prev_fake_probs = tf.placeholder(tf.float32, shape=[self.batch_size])

        self.d_loss = tf.reduce_mean(-tf.log(self.discriminator_real[:, :self.g_index + 1, :])
                                     - tf.log(1 - self.discriminator_fake[:, :self.g_index + 1, :]))

        mask = tf.one_hot(tf.squeeze(self.fake_data[:, self.g_index]), depth=INPUT_SIZE, dtype=tf.bool, on_value=True, off_value=False,
                          axis=-1)
        self.chosen_probs = tf.boolean_mask(tf.squeeze(self.generated[:, self.g_index, :]), mask)

        self.fake_probs = self.discriminator_fake[:, self.g_index, :]
        diff = tf.squeeze(self.fake_probs - tf.expand_dims(self.prev_fake_probs, axis=1))
        mean, variance = tf.nn.moments(diff, axes=[0])
        self.reward = (diff - mean) / (tf.sqrt(variance) + 0.000001)

        policy_loss = -tf.reduce_mean(tf.multiply(tf.log(self.chosen_probs), self.reward))
        self.avg_disc_fake = tf.reduce_mean(self.discriminator_fake)
        self.avg_disc_real = tf.reduce_mean(self.discriminator_real)
        entropy_loss = -tf.reduce_mean(self.entropy(tf.squeeze(self.generated)))
        self.g_loss = policy_loss + 2 * entropy_loss

        self.op_d = self.optimizer(self.d_loss, d_params)
        self.op_g = self.optimizer(self.g_loss, g_params)

        self.session = tf.Session()
        self.summary_disc_loss = tf.placeholder(tf.float32, shape=[])
        self.summary_gen_loss = tf.placeholder(tf.float32, shape=[])

        tf.summary.scalar('avg_fake_prob', self.summary_disc_loss)
        tf.summary.scalar('avg_real_prob', self.summary_gen_loss)

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

    def generator(self, rnn_input, initial_states, noise, index):
        """
        :param rnn_input: A tensor with shape [batch_size, num_Steps, input_size]
        :param initial_states: A tuple of the initial state for each rnn
        :param noise: A tensor with shape [batch_size, noise_dimension] concatenated to each input
        :return: A tensor with shape [batch_size, num_steps, output_size]
        """
        # TODO: Add in noise

        output, state = tf.nn.dynamic_rnn(self.generator_cell, rnn_input, initial_state=initial_states)
        output = self.generator_dense(output)

        # self.gen_final_output = output[:, index, :]
        self.g_final_state = state
        return output

    def generator_single(self, rnn_input, initial_states, noise):
        """
        :param rnn_input: Tensor with shape [batch_size, input_size]
        :param initial_states: Tuple of initial states for each rnn
        :param noise: Tensor with shape [batch_size, noise_dimension]
        :return: Tensor with shape [batch_size, 1, output_size]
        """
        output, state = self.generator_cell(rnn_input, initial_states)
        output = tf.expand_dims(output, axis=1)
        output = self.generator_dense(output)
        self.gen_final_output_single = output[:, -1, :]
        self.g_final_state_single = state
        return output

    def generator_dense(self, input_tensor):
        """
        :param input_tensor: Tensor of shape [batch_size, num_steps, rnn_size]
        :return: Tensor of shape [batch_size, num_steps, input_size]
        """
        weights_out = tf.get_variable("out_layer_w", initializer=tf.contrib.layers.xavier_initializer(),
                                      shape=[self.rnn_size, INPUT_SIZE], dtype=tf.float32)
        bias_out = tf.get_variable("out_layer_b", initializer=tf.zeros_initializer, shape=[INPUT_SIZE])
        return tf.nn.softmax(tf.tensordot(input_tensor, weights_out, [[2], [0]]) + bias_out)

    def discriminator(self, rnn_input, initial_state):
        """
        :param initial_state: The initial state of the rnn
        :param rnn_input: A tensor with shape [batch_size, num_steps, input_size]
        :return: A tensor with shape [batch_size, num_steps, 1]
        """
        output, final_state = tf.nn.dynamic_rnn(self.discriminator_cell, rnn_input, initial_state=initial_state)

        weights_out = tf.get_variable("out_layer_w", initializer=tf.contrib.layers.xavier_initializer(),
                                      shape=[self.rnn_size_d, 1], dtype=tf.float32)
        bias_out = tf.get_variable("out_layer_b", initializer=tf.zeros_initializer, shape=[1])
        output = tf.tensordot(output, weights_out, [[2], [0]]) + bias_out
        output = tf.nn.sigmoid(output)

        return output, final_state

    def sample_generator(self, iteration):
        batches = np.zeros((self.batch_size, 0, INPUT_SIZE))

        previous_output = [Util.CONTINUE_SYMBOL] * self.batch_size
        state_input = np.zeros((self.num_rnn_layers, self.batch_size, self.rnn_size))
        noise_input = self.fixed_sample_noise

        piano_roll = []
        curr_timestep = []

        for i in range(250):
            batch, probs, state_input = self.session.run([
                self.generated_single, self.gen_final_output_single, self.g_final_state_single
            ], feed_dict={
                self.input_gen_single: previous_output,
                self.g_state: state_input,
                self.g_noise: noise_input
            })
            batches = np.append(batches, batch, axis=1)

            note_index = Util.sample(probs[0])
            previous_output = [note_index] * self.batch_size

            if Util.is_continue_symbol(note_index):
                piano_roll.append(curr_timestep)
                curr_timestep = []
            else:
                curr_timestep.append(Util.index_to_midi(note_index))

        if len(curr_timestep) != 0:
            piano_roll.append(curr_timestep)

        prefix = './samples/1/'
        if not os.path.exists(prefix):
            os.makedirs(prefix)

        filename = prefix + ("%05d" % (iteration,))
        with open(filename + '.txt', 'w') as f:
            for item in piano_roll:
                f.write("%s\n" % str(item))
        print("Saving midi file " + filename)
        Util.piano_roll_to_midi(piano_roll, filename)

    @staticmethod
    def entropy(probabilities):
        return -tf.reduce_sum(probabilities * tf.log(probabilities), 1, name='entropy')

    @staticmethod
    def optimizer(loss, var_list):
        return tf.train.AdamOptimizer(0.0007, beta1=0.5).minimize(loss, var_list=var_list)

    def train(self):
        total_loss_d = total_loss_g = total_avg_d_fake = total_avg_d_real = counter = 0
        trained_d = trained_g = 0
        avg_d_fake = avg_d_real = 0.0
        g_index = 0

        num_batches = int(np.size(self.data) / (self.num_steps * self.batch_size))
        num_epochs = 500

        for epoch in range(num_epochs):
            self.sample_generator(epoch)

            for batch_num in range(num_batches):
                real_data = self.data[:, batch_num * self.num_steps: (batch_num + 1) * self.num_steps]

                # Go through n batches before resetting state
                if batch_num % 64 == 0:
                    g_noise = np.random.normal(size=(self.batch_size, self.noise_dimension))
                    g_state = np.zeros((self.num_rnn_layers, self.batch_size, self.rnn_size))
                    d_state_fake = np.zeros((self.num_rnn_layers_d, self.batch_size, self.rnn_size_d))
                    d_state_real = np.zeros((self.num_rnn_layers_d, self.batch_size, self.rnn_size_d))
                    gen_input = np.empty((self.batch_size, self.num_steps), dtype=np.int32)
                    gen_input.fill(Util.CONTINUE_SYMBOL)
                    prev_fake_probs = np.empty((self.batch_size))
                    prev_fake_probs.fill(0.5)
                    g_index = 0

                # To train:
                # Step 1: Run generator for single step to get distribution. Record old_state, new_state, old_input.
                generated, g_state_new = self.session.run([self.generated, self.g_final_state], feed_dict={
                    self.g_state: g_state,
                    self.input_gen: gen_input,
                    self.g_noise: g_noise,
                    self.g_index: g_index
                })

                # Step 2: Sample from the distribution to get action, record new_input.
                # Needs to be shape [batch_size, 1]
                g_new_output = np.reshape(Util.sample_multiple(generated[:, g_index, :]), newshape=[-1])
                fake_data = np.roll(gen_input, shift=-1, axis=-1)
                fake_data[:, g_index] = g_new_output

                # Step 3: Run (and train) discriminator on this new note and get reward.
                feed_dict = {
                    self.d_state_fake: d_state_fake,
                    self.d_state_real: d_state_real,

                    self.g_state: g_state,
                    self.input_gen: gen_input,
                    self.g_noise: g_noise,

                    self.fake_data: fake_data,
                    self.real_data: real_data,
                    self.g_index: g_index,

                    self.prev_fake_probs: prev_fake_probs
                }

                ops = [self.d_loss, self.g_loss, self.g_final_state, self.d_final_state_fake, self.d_final_state_real,
                       self.discriminator_fake, self.fake_probs, self.chosen_probs,
                       self.avg_disc_fake, self.avg_disc_real, self.reward]

                if batch_num % 5 == 0 and avg_d_fake - avg_d_real > -0.1:
                    ops.append(self.op_d)
                    trained_d += 1

                # if avg_d_fake < 0.6:
                ops.append(self.op_g)
                trained_g += 1

                vals = self.session.run(ops, feed_dict=feed_dict)
                loss_d, loss_g, state_new, d_state_fake, d_state_real, d_fake, fake_probs, chosen_probs,\
                    avg_d_fake, avg_d_real, reward, = vals[:11]
                prev_fake_probs = np.squeeze(fake_probs)

                # Step 4: Reassign state and prev_output
                g_state = g_state_new
                if g_index == self.num_steps - 1:
                    gen_input = fake_data
                else:
                    gen_input[:, g_index + 1] = g_new_output
                # gen_input = np.roll(gen_input, shift=-1, axis=-1)
                # gen_input[:, -1] = g_new_output

                total_loss_d += loss_d
                total_loss_g += loss_g
                total_avg_d_fake += avg_d_fake
                total_avg_d_real += avg_d_real
                counter += 1
                g_index = min(g_index + 1, self.num_steps - 1)

                if (batch_num + 1) % 100 == 0 or batch_num == num_batches - 1:
                    print("Batch %04d/%d \t D: %.3f \t G: %.3f \t Fake: %.3f \t Real: %.3f \t G: %.0f%% \t D: %.0f%%" %
                          (batch_num + 1, num_batches,
                           total_loss_d / counter,
                           total_loss_g / counter,
                           total_avg_d_fake / counter,
                           total_avg_d_real / counter,
                           trained_g / counter * 100,
                           trained_d / counter * 100
                           ))
                    total_loss_d = total_loss_g = total_avg_d_fake = total_avg_d_real = counter = 0
                    trained_g = trained_d = 0

            # self.save_summaries(total_avg_d_fake / counter, total_avg_d_real / counter, epoch)
            total_loss_d = total_loss_g = total_avg_d_fake = total_avg_d_real = counter = 0
            trained_g = trained_d = 0


if __name__ == '__main__':
    run_name = sys.argv[1]
    learner = Learner('./data/MuseData.pickle', run_name)
    learner.train()
