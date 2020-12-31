import numpy as np
from gensim.models import Word2Vec
from tensorflow.python.keras.utils.data_utils import Sequence

from w2v_test.costants import IMAGE_SIZE, CONTEXT_LENGTH, PLACEHOLDER, BATCH_SIZE
from utils.utils import get_preprocessed_img, get_token_from_gui


class DataGenerator(Sequence):
    'Generates data for Keras'

    def __init__(self, img_paths, gui_paths, word_model: Word2Vec, shuffle=True, batch_size=BATCH_SIZE):
        'Initialization'
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.word_model = word_model
        self.samples = self.create_samples(img_paths, gui_paths)
        self.on_epoch_end()

    def context_to_w2v_indexes(self, partial_sequences):
        temp = []
        for sequence in partial_sequences:
            sparse_vectors_sequence = []
            for token in sequence:
                sparse_vectors_sequence.append(self.word_model.wv.vocab[token].index)
            temp.append(np.array(sparse_vectors_sequence))

        return temp

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.samples) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        tmp_samples = [self.samples[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(tmp_samples)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def create_samples(self, img_paths, gui_paths):

        images = []
        contexts = []
        labels = []
        # Generate data
        for i in range(0, len(gui_paths)):
            img = get_preprocessed_img(img_paths[i], IMAGE_SIZE)

            gui = open(gui_paths[i], 'r')

            token_sequence = get_token_from_gui(gui)

            suffix = [PLACEHOLDER] * CONTEXT_LENGTH

            a = np.concatenate([suffix, token_sequence])
            for j in range(0, len(a) - CONTEXT_LENGTH):
                context = a[j:j + CONTEXT_LENGTH]
                label = a[j + CONTEXT_LENGTH]  # label = name
                encoded_label = self.word_model.wv.vocab[label].index

                contexts.append(context)
                images.append(img)
                labels.append(encoded_label)

        contexts = self.context_to_w2v_indexes(contexts)
        assert len(images) == len(contexts) == len(labels)

        samples = [{'img': images[i], 'context': contexts[i], 'label': labels[i]} for i in range(len(images))]

        return samples

    def __data_generation(self, tmp_samples):
        'Generates data containing batch_size samples'

        # Initialization
        batch_input_images = []
        batch_partial_sequences = []
        batch_next_words = []

        # Generate data
        for i in range(0, len(tmp_samples)):
            batch_input_images.append(tmp_samples[i]['img'])
            batch_partial_sequences.append(tmp_samples[i]['context'])
            batch_next_words.append(tmp_samples[i]['label'])

        batch_input_images = np.array(batch_input_images)
        batch_partial_sequences = np.array(batch_partial_sequences)
        batch_next_words = np.array(batch_next_words)

        return [batch_input_images, batch_partial_sequences], batch_next_words
