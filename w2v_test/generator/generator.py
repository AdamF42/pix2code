import numpy as np
from gensim.models import Word2Vec
from tensorflow.python.keras.utils.data_utils import Sequence

CONTEXT_LENGTH = 48
IMAGE_SIZE = 256
BATCH_SIZE = 64
START_TOKEN = "<START>"
END_TOKEN = "<END>"
PLACEHOLDER = " "


class DataGenerator(Sequence):
    'Generates data for Keras'

    def __init__(self, img_paths, labels, word_model: Word2Vec, batch_size=64):
        'Initialization'
        self.batch_size = batch_size
        self.labels = labels
        self.img_paths = img_paths
        self.on_epoch_end()
        self.word_model = word_model

    def indexify(self, partial_sequences):
        temp = []
        for sequence in partial_sequences:
            sparse_vectors_sequence = []
            for token in sequence:
                sparse_vectors_sequence.append(self.word_model.wv.vocab[token].index)
            temp.append(np.array(sparse_vectors_sequence))

        return temp

    @staticmethod
    def get_preprocessed_img(img_path, image_size):
        import cv2
        img = cv2.imread(img_path)
        img = cv2.resize(img, (image_size, image_size))
        img = img.astype('float32')
        img /= 255
        return img

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.img_paths) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        img_paths_temp = [self.img_paths[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(img_paths_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.img_paths))
        # print("GENERATOR INDEX: {}",format(self.indexes))
        # if self.shuffle == True:
        #     np.random.shuffle(self.indexes)

    def __data_generation(self, gui_paths_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)

        # Initialization
        batch_input_images = []
        batch_partial_sequences = []
        batch_next_words = []
        sample_in_batch_counter = 0

        # Generate data
        for i in range(0, len(gui_paths_temp)):
            # if self.gui_paths[i].find(".png") != -1: TODO check
            img = self.get_preprocessed_img(self.img_paths[i], IMAGE_SIZE)
            gui = open(self.labels[i], 'r')

            token_sequence = [START_TOKEN]
            for line in gui:
                line = line.replace(" ", "  ").replace(",", " ,").replace("\n", " \n")
                tokens = line.split(" ")
                tokens = map(lambda x: " " if x == "" else x, tokens)
                for token in tokens:
                    token_sequence.append(token)
            token_sequence.append(END_TOKEN)

            suffix = [PLACEHOLDER] * CONTEXT_LENGTH

            a = np.concatenate([suffix, token_sequence])
            for j in range(0, len(a) - CONTEXT_LENGTH):
                context = a[j:j + CONTEXT_LENGTH]
                label = a[j + CONTEXT_LENGTH]  # label = name

                batch_input_images.append(img)
                batch_partial_sequences.append(context)

                encoding = self.word_model.wv.vocab[label].index

                batch_next_words.append(encoding)
                sample_in_batch_counter += 1

                if sample_in_batch_counter == self.batch_size:
                    # if verbose:
                    # print("Generating sparse vectors...")
                    batch_partial_sequences = self.indexify(batch_partial_sequences)

                    batch_input_images = np.array(batch_input_images)  # ndarray -> shape (64, 256, 256, 3)
                    batch_partial_sequences = np.array(batch_partial_sequences)  # ndarray -> shape (64, 48, 19)
                    batch_next_words = np.array(batch_next_words)  # ndarray -> shape (64, 19)

                    print([batch_input_images.shape, batch_partial_sequences.shape], batch_next_words.shape)

                    return [batch_input_images, batch_partial_sequences], batch_next_words
