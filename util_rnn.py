import numpy as np
import midi
import random

MIDI_START = 21
MIDI_END = 108


class Util:
    CONTINUE_SYMBOL = MIDI_END - MIDI_START + 1

    @staticmethod
    def sample_multiple(probabilities):
        """
        Treats each row of a 2d matrix as a probability distribution and returns sampled indices
        :param probabilities: predictions with shape [batch_size, output_size]
        :return: sampled indices with shape [batch_size]
        """
        cumulative = probabilities.cumsum(axis=1)
        uniform = np.random.rand(len(cumulative), 1)
        choices = (uniform < cumulative).argmax(axis=1)
        return choices

    @staticmethod
    def sample(predictions, temperature=1.0):
        """
        :param predictions: A list of floats representing probabilities
        :param temperature: The temperature of the softmax, lower values makes it closer to hard max
        :return: An index of the predictions
        """
        predictions = np.asarray(predictions).astype('float64')
        predictions = np.log(predictions) / temperature
        exp_preds = np.exp(predictions)
        predictions = exp_preds / np.sum(exp_preds)
        probabilities = np.random.multinomial(1, predictions, 1)
        return np.argmax(probabilities)

    @staticmethod
    def is_continue_symbol(symbol):
        return symbol == MIDI_END - MIDI_START + 1

    @staticmethod
    def index_to_midi(index):
        value = index + MIDI_START
        if value < MIDI_START or value > MIDI_END:
            raise Exception("Midi number " + str(value) + " out of bounds.")
        return value

    @staticmethod
    def piano_roll_to_midi(piano_roll, filename):
        """
        Saves a midi file from a piano_roll object

        :param piano_roll: A list of lists, where each sublist represents a time step and contains
            the midi numbers
        :param filename: The file name of the output file
        :return: The midi track that was created
        """
        pattern = midi.Pattern()
        track = midi.Track()
        pattern.append(track)
        offset = 0

        for time_step, notes in enumerate(piano_roll):
            for note in notes:
                previous_notes = piano_roll[time_step - 1] if time_step > 0 else []
                if note not in previous_notes:
                    track.append(midi.NoteOnEvent(tick=offset, velocity=100, pitch=note))
                    offset = 0
            offset += 130
            for note in notes:
                next_notes = piano_roll[time_step + 1] if time_step < len(piano_roll) - 1 else []
                if note not in next_notes:
                    track.append(midi.NoteOffEvent(tick=offset, pitch=note))
                    offset = 0

        eot = midi.EndOfTrackEvent(tick=1)
        track.append(eot)
        midi.write_midifile(filename + '.mid', pattern)
        return track


class DataHandler:
    def __init__(self):
        pass

    @staticmethod
    def preprocess(dataset, batch_size):
        """
        :param batch_size: Size of the batches for reshaping
        :param dataset: The dataset from the pickle file
        :return: A continuous stream of indices, with a special number representing next time step,
            reshaped such that each row is a sequence
        """
        data = dataset['test'] + dataset['train'] + dataset['valid']

        # Concatenate all music into one array
        continuous = []
        for piece in data:
            continuous += piece
            # continuous += [[] for _ in range(4)]

        data = []
        for time_step in continuous:
            random.shuffle(time_step)
            for note in time_step:  # sorted(time_step, reverse=True):
                data.append(note - MIDI_START)
            data.append(MIDI_END - MIDI_START + 1)

        sequence_length = int(len(data) / batch_size)
        return np.reshape(data[:sequence_length * batch_size], (batch_size, sequence_length))

        # num_batches = int(len(data) / (batch_size * num_steps))
        # return np.reshape(data[:num_batches * batch_size * num_steps], (batch_size, num_batches * num_steps))
