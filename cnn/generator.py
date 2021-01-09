import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils.data_utils import Sequence
from tqdm import tqdm
import tensorflow as tf
from utils.costants import BATCH_SIZE, IMAGE_SIZE
from utils.utils import get_preprocessed_img, get_token_from_gui, get_output_names


class DataGenerator(Sequence):
    'Generates data for Keras'

    def __init__(self, img_paths, gui_paths, output_names, tokens_to_exclude,
                 samples=None, shuffle=True, code_encoder=None,
                 image_transformer=None, batch_size=BATCH_SIZE):
        'Initialization'
        self.output_names = output_names
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.samples = samples
        self.tokens_to_exclude = tokens_to_exclude
        self.code_encoder=code_encoder
        self.image_transformer = image_transformer

        if samples is None:
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
        for i in tqdm(range(0, len(gui_paths)), desc="Creating samples"):
            # for i in range(0, len(gui_paths)):
            img = get_preprocessed_img(img_paths[i], IMAGE_SIZE)

            gui = open(gui_paths[i], 'r')
            token_sequence = get_token_from_gui(gui, self.tokens_to_exclude)
            token_sequence = get_output_names(token_sequence)

            tokens_count = {}
            for name in get_output_names(self.output_names):
                tokens_count[name + '_count'] = 0

            for token in token_sequence:
                tokens_count[token + "_count"] = tokens_count[token + "_count"] + 1

            if self.code_encoder:
                code_array = self.code_encoder(token_sequence)
                tokens_count.update({'code': tf.convert_to_tensor(code_array)})
            # tokens_count.update({'code': token_sequence})
            labels.append(tokens_count)
            images.append(img)

        samples = [{'img': images[i], 'label': labels[i]} for i in range(len(images))]
        return samples

    def __data_generation(self, tmp_samples):
        'Generates data containing batch_size samples'

        # Initialization
        images = []
        labels = []

        # Generate data
        for i in range(0, len(tmp_samples)):
            images.append(tmp_samples[i]['img'])
            labels.append(tmp_samples[i]['label'])

        labels_dict = {}
        for dict in labels:
            for key, val in dict.items():
                if labels_dict.get(key) is None:
                    labels_dict[key] = []
                labels_dict[key].append(val)
                # labels_dict.update({key: val})

        for key in labels_dict.keys():
            labels_dict[key] = np.array(labels_dict[key])

        if self.image_transformer:
            images_dict = {'img_data': np.array([self.image_transformer.random_transform(img) for img in images])}
        else:
            images_dict = {'img_data': np.array([img for img in images])}

        return images_dict, labels_dict
