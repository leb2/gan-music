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

        self.batch_size = 512
        self.num_steps = 1
        self.data = DataHandler.preprocess(self.dataset, self.batch_size)

        print("Initializing...")
        self.rnn_size = 128
        self.rnn_size_d = 128

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

        print("Initializing generator")
        with tf.variable_scope('G'):

            # -- GENERATION -- #
            self.input_gen = tf.placeholder(tf.int32, shape=[self.batch_size], name="input")
            self.g_state = tf.placeholder(tf.float32, [self.num_rnn_layers, self.batch_size, self.rnn_size],
                                          name="state")
            self.g_noise = tf.placeholder(tf.float32, [self.batch_size, self.noise_dimension], name="noise")

            g_state = tuple([self.g_state[i] for i in range(self.num_rnn_layers)])

            # Shape [batch_size, 1, INPUT_SIZE]
            input_gen = tf.one_hot(self.input_gen, depth=INPUT_SIZE, axis=-1)
            self.generated = self.generator(input_gen, g_state, self.g_noise)

        print("Initializing discriminator")
        self.embedding = tf.Variable(tf.random_normal([INPUT_SIZE, self.embed_size], stddev=0.01))
        self.real_data = tf.placeholder(tf.int32, shape=[self.batch_size, self.num_steps])
        self.fake_data = tf.placeholder(tf.int32, shape=[self.batch_size, self.num_steps])
        self.d_state_real = tf.placeholder(tf.float32, [self.num_rnn_layers_d, self.batch_size, self.rnn_size_d])
        self.d_state_fake = tf.placeholder(tf.float32, [self.num_rnn_layers_d, self.batch_size, self.rnn_size_d])

        with tf.variable_scope('D') as scope:
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
        self.d_loss = tf.reduce_mean(-tf.log(self.discriminator_real[:, :, :])
                                     - tf.log(1 - self.discriminator_fake[:, :, :]))

        mask = tf.one_hot(tf.squeeze(self.fake_data), depth=INPUT_SIZE, dtype=tf.bool, on_value=True, off_value=False,
                          axis=-1)
        self.chosen_probs = tf.boolean_mask(tf.squeeze(self.generated), mask)

        policy_loss = -tf.reduce_mean(tf.multiply(tf.log(self.chosen_probs), tf.squeeze(self.discriminator_fake - 0.5)))
        self.avg_disc_fake = tf.reduce_mean(self.discriminator_fake)
        self.avg_disc_real = tf.reduce_mean(self.discriminator_real)
        entropy_loss = -tf.reduce_mean(self.entropy(tf.squeeze(self.generated)))
        self.g_loss = policy_loss + 0.01 * entropy_loss

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
        output = tf.layers.dense(output, INPUT_SIZE, name='output_layer', activation=tf.nn.softmax)

        self.gen_final_output = output
        self.g_final_state = state
        return tf.expand_dims(output, axis=1)

    def discriminator(self, rnn_input, initial_state):
        """
        :param initial_state: The initial state of the rnn
        :param rnn_input: A tensor with shape [batch_size, num_steps, input_size]
        :return: A tensor with shape [batch_size, 1, 1]
        """
        discriminator = tf.contrib.rnn.MultiRNNCell([
            tf.contrib.rnn.GRUCell(self.rnn_size_d) for _ in range(self.num_rnn_layers_d)
        ])
        output, final_state = tf.nn.dynamic_rnn(discriminator, rnn_input, initial_state=initial_state)

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
                self.generated, self.gen_final_output, self.g_final_state
            ], feed_dict={
                self.input_gen: previous_output,
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
        return tf.train.AdamOptimizer(0.003).minimize(loss, var_list=var_list)

    def train(self):
        total_loss_d = total_loss_g = total_avg_d_fake = total_avg_d_real = counter = 0
        trained_d = trained_g = 0
        num_batches = int(np.size(self.data) / (self.num_steps * self.batch_size))
        num_epochs = 500

        g_noise = np.random.normal(size=(self.batch_size, self.noise_dimension))
        g_state = np.zeros((self.num_rnn_layers, self.batch_size, self.rnn_size))

        d_state_fake = np.zeros((self.num_rnn_layers_d, self.batch_size, self.rnn_size_d))
        d_state_real = np.zeros((self.num_rnn_layers_d, self.batch_size, self.rnn_size_d))

        g_prev_output = Util.CONTINUE_SYMBOL
        avg_d_fake = avg_d_real = 0.0

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
                    g_prev_output = [Util.CONTINUE_SYMBOL] * self.batch_size

                # To train:
                # Step 1: Run generator for single step to get distribution. Record old_state, new_state, old_input.
                generated, g_state_new = self.session.run([self.generated, self.g_final_state], feed_dict={
                    self.input_gen: g_prev_output,
                    self.g_state: g_state,
                    self.g_noise: g_noise
                })

                # Step 2: Sample from the distribution to get action, record new_input.
                # Needs to be shape [batch_size, input_size]
                g_new_output = np.reshape(Util.sample_multiple(generated[:, 0, :]), newshape=[-1, 1])

                # Step 3: Run (and train) discriminator on this new note and get reward.
                feed_dict = {
                    self.d_state_fake: d_state_fake,
                    self.d_state_real: d_state_real,
                    self.fake_data: g_new_output,
                    self.real_data: real_data,
                    self.g_state: g_state,
                    self.input_gen: g_prev_output,
                    self.g_noise: g_noise
                }
                ops = [self.d_loss, self.g_loss, self.g_final_state, self.d_final_state_fake, self.d_final_state_real,
                       self.discriminator_fake, self.chosen_probs, self.avg_disc_fake, self.avg_disc_real]

                if avg_d_fake - avg_d_real > -0.1:
                    ops.append(self.op_d)
                    trained_d += 1

                if avg_d_fake < 0.6:
                    ops.append(self.op_g)
                    trained_g += 1

                vals = self.session.run(ops, feed_dict=feed_dict)
                loss_d, loss_g, state_new, d_state_fake, d_state_real, d_fake, chosen_probs,\
                    avg_d_fake, avg_d_real = vals[:9]

                # Step 4: Reassign state and prev_output
                g_state = g_state_new
                g_prev_output = np.squeeze(g_new_output)

                total_loss_d += loss_d
                total_loss_g += loss_g
                total_avg_d_fake += avg_d_fake
                total_avg_d_real += avg_d_real
                counter += 1

                if (batch_num + 1) % 500 == 0 or batch_num == num_batches - 1:
                    print("Batch %04d/%d \t D: %.3f \t G: %.3f \t Fake: %.3f \t Real: %.3f \t G: %.1f%% \t D: %.1f%%" %
                          (batch_num + 1, num_batches,
                           total_loss_d / counter,
                           total_loss_g / counter,
                           total_avg_d_fake / counter,
                           total_avg_d_real / counter,
                           trained_g / counter * 100,
                           trained_d / counter * 100
                           ))

            self.save_summaries(total_avg_d_fake / counter, total_avg_d_real / counter, epoch)
            total_loss_d = total_loss_g = total_avg_d_fake = total_avg_d_real = counter = 0
            trained_g = trained_d = 0


if __name__ == '__main__':
    run_name = sys.argv[1]
    learner = Learner('./data/MuseData.pickle', run_name)
    learner.train()
