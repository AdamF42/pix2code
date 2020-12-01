import os

import numpy as np
from gensim.models import Word2Vec

from w2v_test.costants import START_TOKEN, END_TOKEN, PLACEHOLDER, CONTEXT_LENGTH, IMAGE_SIZE
from w2v_test.dataset.utils import get_preprocessed_img, show, get_token_from_gui


class Dataset:
    def __init__(self, word_model: Word2Vec):
        self.input_shape = None
        self.output_size = None

        self.word_model = word_model

        self.ids = []
        self.input_images = []
        self.partial_sequences = []
        self.next_words = []

        self.size = 0
        self.data = []

    @staticmethod
    def load_paths_only(path):
        print("Loading paths...")
        gui_paths = []
        img_paths = []
        for f in os.listdir(path):
            if f.find(".gui") != -1:
                path_gui = "{}/{}".format(path, f)
                gui_paths.append(path_gui)
                file_name = f[:f.find(".gui")]

                if os.path.isfile("{}/{}.png".format(path, file_name)):
                    path_img = "{}/{}.png".format(path, file_name)
                    img_paths.append(path_img)
                elif os.path.isfile("{}/{}.npz".format(path, file_name)):
                    path_img = "{}/{}.npz".format(path, file_name)
                    img_paths.append(path_img)

        assert len(gui_paths) == len(img_paths)
        return gui_paths, img_paths

    def load_with_word2vec(self, path):
        self.load(path)
        print("Generating sparse vectors w2v...")
        self.create_word2vec_representation()

    def load(self, path):
        print("Loading data...")
        for f in os.listdir(path):
            if f.find(".gui") != -1:

                gui = open("{}/{}".format(path, f), 'r')
                file_name = f[:f.find(".gui")]

                if os.path.isfile("{}/{}.png".format(path, file_name)):
                    img = get_preprocessed_img("{}/{}.png".format(path, file_name), IMAGE_SIZE)
                    self.append(file_name, gui, img)

                elif os.path.isfile("{}/{}.npz".format(path, file_name)):
                    img = np.load("{}/{}.npz".format(path, file_name))["features"]
                    self.append(file_name, gui, img)
        self.input_shape = self.input_images[0].shape

    def convert_arrays(self):
        print("Convert arrays into np.array...")
        self.input_images = np.array(self.input_images)
        self.partial_sequences = np.array(self.partial_sequences)
        self.next_words = np.array(self.next_words)

    def append(self, sample_id, gui, img, to_show=False):
        self.size += 1

        if to_show:
            pic = img * 255
            pic = np.array(pic, dtype=np.uint8)
            show(pic)

        token_sequence = get_token_from_gui(gui)

        suffix = [PLACEHOLDER] * CONTEXT_LENGTH

        a = np.concatenate([suffix, token_sequence])
        for j in range(0, len(a) - CONTEXT_LENGTH):
            context = a[j:j + CONTEXT_LENGTH]
            label = a[j + CONTEXT_LENGTH]
            self.ids.append(sample_id)
            self.input_images.append(img)
            self.partial_sequences.append(context)
            self.next_words.append(label)

    def word2idx(self, word: str):
        return self.word_model.wv.vocab[word].index

    def idx2word(self, idx: int):
        return self.word_model.wv.index2word[idx]

    def create_word2vec_representation(self):

        print(len(self.input_images))
        print(len(self.partial_sequences))
        print(len(self.next_words))

        assert len(self.input_images) == len(self.partial_sequences) == len(self.next_words)

        self.next_words = self.w2v_encode_next_words()

        self.partial_sequences = self.w2v_encode_partial_sequences()

        self.convert_arrays()

    def w2v_encode_partial_sequences(self):
        temp = []
        for sequence in self.partial_sequences:
            sparse_vectors_sequence = []
            for token in sequence:
                sparse_vectors_sequence.append(self.word2idx(token))
            temp.append(np.array(sparse_vectors_sequence))

        return temp

    def w2v_encode_next_words(self):
        temp = []
        for token in self.next_words:
            temp.append(self.word2idx(token))
        return temp
