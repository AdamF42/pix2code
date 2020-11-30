from __future__ import print_function

__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

import os

import numpy as np
from gensim.models import Word2Vec


CONTEXT_LENGTH = 48
IMAGE_SIZE = 256
BATCH_SIZE = 64
START_TOKEN = "<START>"
END_TOKEN = "<END>"
PLACEHOLDER = " "

class Utils:
    @staticmethod
    def sparsify(label_vector, output_size):
        sparse_vector = []

        for label in label_vector:
            sparse_label = np.zeros(output_size)
            sparse_label[label] = 1

            sparse_vector.append(sparse_label)

        return np.array(sparse_vector)

    @staticmethod
    def get_preprocessed_img(img_path, image_size = IMAGE_SIZE):
        import cv2
        img = cv2.imread(img_path)
        img = cv2.resize(img, (image_size, image_size))
        img = img.astype('float32')
        img /= 255
        return img

    @staticmethod
    def show(image):
        import cv2
        cv2.namedWindow("view", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("view", image)
        cv2.waitKey(0)
        cv2.destroyWindow("view")

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
        print("Parsing data...")
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

    def load_with_word2vec(self, path, generate_binary_sequences=False):
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
                    img = Utils.get_preprocessed_img("{}/{}.png".format(path, file_name), IMAGE_SIZE)
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
            Utils.show(pic)

        token_sequence = [START_TOKEN]
        for line in gui:
            line = line.replace(" ", "  ") \
                .replace(",", " ,") \
                .replace("\n", " \n") \
                .replace("{", " { ") \
                .replace("}", " } ") \
                .replace(",", " , ")
            tokens = line.split(" ")
            tokens = map(lambda x: " " if x == "" else x, tokens)
            for token in tokens:
                token_sequence.append(token)
        token_sequence.append(END_TOKEN)

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

        # for i in range(0, self.size):
        #     self.next_words[i] = self.word2idx(self.next_words[i])
        self.next_words = self.w2v_encode_next_words()

        self.partial_sequences =  self.w2v_encode_partial_sequences()

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

