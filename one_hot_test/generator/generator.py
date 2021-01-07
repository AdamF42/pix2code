import numpy as np
from tensorflow.python.keras.utils.data_utils import Sequence
from tqdm import tqdm

from utils.costants import IMAGE_SIZE, CONTEXT_LENGTH, PLACEHOLDER, BATCH_SIZE
from utils.utils import get_preprocessed_img, get_token_from_gui, get_output_names


class DataGenerator(Sequence):
    'Generates data for Keras'

    def __init__(self, img_paths, gui_paths, voc, output_names, shuffle=True, max_code_len=CONTEXT_LENGTH,
                 batch_size=BATCH_SIZE, is_count_required=False, is_with_output_name=False):
        'Initialization'
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.voc = voc
        self.max_code_len = max_code_len
        self.is_with_output_name = is_with_output_name
        self.output_names = output_names
        self.is_count_required = is_count_required
        self.samples = self.create_samples(img_paths, gui_paths, self.voc)
        self.on_epoch_end()

    def context_to_one_hot_indexes(self, partial_sequences):
        temp = []
        for sequence in partial_sequences:
            sparse_vectors_sequence = []
            for token in sequence:
                sparse_vectors_sequence.append(self.voc.word_to_index(token))
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

    def create_samples(self, img_paths, gui_paths, voc):

        images = []
        contexts = []
        labels = []
        # Generate data
        for i in tqdm(range(0, len(gui_paths)), desc="Creating samples"):
            img = get_preprocessed_img(img_paths[i], IMAGE_SIZE)

            gui = open(gui_paths[i], 'r')

            token_sequence = get_token_from_gui(gui)

            if self.is_with_output_name:
                token_sequence = get_output_names(token_sequence)

            suffix = [PLACEHOLDER] * self.max_code_len

            a = np.concatenate([suffix, token_sequence])
            for j in range(0, len(a) - self.max_code_len):
                context = a[j:j + self.max_code_len]
                label = a[j + self.max_code_len]  # label = name
                one_hot_encoding = voc.word_to_index(label)

                if self.is_count_required:
                    tokens_count = {}
                    for name in get_output_names(self.output_names):
                        tokens_count[name + "_count"] = 0

                    for token in token_sequence:
                        tokens_count[token + "_count"] = tokens_count[token + "_count"] + 1

                    tokens_count.update({'code': one_hot_encoding})
                    one_hot_encoding = tokens_count

                contexts.append(context)
                images.append(img)
                labels.append(one_hot_encoding)

        contexts = self.context_to_one_hot_indexes(contexts)

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

        # batch_next_words_dict = {}
        # for dict in batch_next_words:
        #     for k, v in dict.items():
        #         if batch_next_words_dict.get(k) is None:
        #             batch_next_words_dict[k] = []
        #         batch_next_words_dict[k].append(v)
        #
        # for key in batch_next_words_dict.keys():
        #     batch_next_words_dict[key] = np.array(batch_next_words_dict[key])


        batch_input_images = np.array(batch_input_images)
        batch_partial_sequences = np.array(batch_partial_sequences)
        batch_next_words = np.array(batch_next_words)
        # print([batch_input_images.shape, batch_partial_sequences.shape], batch_next_words.shape)
        return [batch_input_images, batch_partial_sequences], batch_next_words
