import numpy as np
import random
import midi
import cPickle

class Util:
    MIDI_START = 21
    MIDI_END = 108
    INPUT_SIZE = MIDI_END - MIDI_START + 1

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
    def index_to_midi(index):
        value = index + Util.MIDI_START
        if value < Util.MIDI_START or value > Util.MIDI_END:
            raise Exception("Midi number " + str(value) + " out of bounds.")
        return value

    @staticmethod
    def midi_to_index(midi):
        if midi < Util.MIDI_START or midi > Util.MIDI_END:
            raise Exception("Midi number " + str(midi) + " out of bounds.")
        return midi - Util.MIDI_START

    @staticmethod
    def tensor_to_midi(name, tensor):
        """
        :param name: Name of file to be created
        :param tensor: A tensor of shape [num_timesteps, 88]
        :return:
        """
        piano_roll = []
        for probabilities in tensor:
            timestep = []
            for index, probability in enumerate(probabilities):
                if random.random() < probability:
                    timestep.append(Util.index_to_midi(index))
            piano_roll.append(timestep)
        Util.piano_roll_to_midi(piano_roll, name)

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
    def generate_data_npy(pickle_path):
        dataset = cPickle.load(file(pickle_path))
        all_pieces = dataset['train'] + dataset['test'] + dataset['valid']
        all_data = []
        for piece in all_pieces:
            all_data += piece
            all_data += [[]] * 8
        print(all_data[0])
        data = np.zeros((len(all_data), Util.INPUT_SIZE))
        for index, timestep in enumerate(all_data):
            for midi in timestep:
                data[index, Util.midi_to_index(midi)] = 1
        print(data.shape)
        np.save('./data/midi.npy', data)
        return data

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
            for note in time_step: # sorted(time_step, reverse=True):
                data.append(note - Util.MIDI_START)
            data.append(Util.MIDI_END - Util.MIDI_START + 1)

        sequence_length = int(len(data) / batch_size)
        return np.reshape(data[:sequence_length * batch_size], (batch_size, sequence_length))

        # num_batches = int(len(data) / (batch_size * num_steps))
        # return np.reshape(data[:num_batches * batch_size * num_steps], (batch_size, num_batches * num_steps))

if __name__ == '__main__':
    DataHandler.generate_data_npy('./data/MuseData.pickle')


