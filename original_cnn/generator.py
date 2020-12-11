import numpy as np
from tensorflow.python.keras.utils.data_utils import Sequence

from w2v_test.costants import IMAGE_SIZE, BATCH_SIZE
from w2v_test.dataset.utils import get_preprocessed_img, get_token_from_gui


class DataGenerator(Sequence):
    'Generates data for Keras'

    def __init__(self, img_paths, gui_paths, output_names, shuffle=True, batch_size=BATCH_SIZE):
        'Initialization'
        self.output_names = output_names
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.samples = self.create_samples(img_paths, gui_paths)
        self.on_epoch_end()

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
        labels = []
        # Generate data
        for i in range(0, len(gui_paths)):
            img = get_preprocessed_img(img_paths[i], IMAGE_SIZE)
            gui = open(gui_paths[i], 'r')
            token_sequence = []
            tokens = filter(lambda x: False if x in ["<START>", "<END>"] else True, get_token_from_gui(gui))
            tokens = map(lambda x: "open_bracket" if x == "{" else x, tokens)
            tokens = map(lambda x: "close_bracket" if x == "}" else x, tokens)
            tokens = map(lambda x: "comma" if x == "," else x, tokens)

            for token in tokens:
                token_sequence.append(token)
            tokens_count = {}
            for name in self.output_names:
                tokens_count[name + '_count'] = 0
            # self.output_names for i in self.output_names]

            for token in token_sequence:
                tokens_count[token + "_count"] = tokens_count[token + "_count"] + 1
                # index = TOKENS_TO_INDEX[token]
                # tokens_count[index] = tokens_count[index] + 1

            # for key in tokens_count.keys():
            #     tokens_count[key] = np.array(tokens_count[key])

            labels.append(tokens_count)
            images.append(img)

        samples = [{'img': images[i], 'label': labels[i]} for i in range(len(images))]
        return samples

    def __data_generation(self, tmp_samples):
        'Generates data containing batch_size samples'

        # Initialization
        batch_input_images = []
        batch_next_words = []

        # Generate data
        for i in range(0, len(tmp_samples)):
            batch_input_images.append(tmp_samples[i]['img'])
            batch_next_words.append(list(tmp_samples[i]['label'].values()))
            # batch_next_words.append(tmp_samples[i]['label'])

        batch_input_images = np.array(batch_input_images)
        batch_next_words = np.array(batch_next_words)

        return batch_input_images, batch_next_words
