from __future__ import print_function

__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

import os

import gensim
import numpy as np
from classes.Utils import *
from classes.Vocabulary import *
from classes.model.Config import *
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class Dataset:
    def __init__(self):
        self.input_shape = None
        self.output_size = None

        self.ids = []
        self.input_images = []
        self.partial_sequences = []
        self.next_words = []

        self.voc = Vocabulary()
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

    def load_with_one_hot_encoding(self, path, generate_binary_sequences=False):
        self.load(path)
        print("Generating sparse vectors...")
        self.voc.create_binary_representation()
        self.create_labeling(generate_binary_sequences)

    def load_with_word2vec(self, path, generate_binary_sequences=False):
        self.load(path)
        print("Generating sparse vectors w2v...")
        self.create_word2vec_representation()
        self.create_labeling(generate_binary_sequences)

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

    def create_labeling(self, generate_binary_sequences):
        self.next_words = self.sparsify_labels(self.next_words, self.voc)
        if generate_binary_sequences:
            self.partial_sequences = self.binarize(self.partial_sequences, self.voc)
        else:
            self.partial_sequences = self.indexify(self.partial_sequences, self.voc)
        self.size = len(self.ids)
        assert self.size == len(self.input_images) == len(self.partial_sequences) == len(self.next_words)
        assert self.voc.size == len(self.voc.vocabulary)
        print("Dataset size: {}".format(self.size))
        print("Vocabulary size: {}".format(self.voc.size))
        self.input_shape = self.input_images[0].shape
        self.output_size = self.voc.size
        print("Input shape: {}".format(self.input_shape))
        print("Output size: {}".format(self.output_size))

    def create_word2vec_representation(self):
        print("Creating w2c representation...")
        w2v_model = gensim.models.Word2Vec(min_count=1,
                                        size=10,
                                        window=3,
                                        # seed=42,
                                        # workers=1,
                                        batch_words=64)
        w2v_model.build_vocab([self.voc.vocabulary])

        # w2v_model.train(self.data, total_examples=len(self.data), epochs=w2v_model.iter)
        w2v_model.train(self.data, total_examples=len(self.data), epochs=3)
        w2v_model.init_sims(replace=True)

        # words = list(w2v_model.wv.vocab)
        # print(words)
        # print(len(words))
        self.tsne_plot(w2v_model)

        for token in self.voc.vocabulary:
            vector = w2v_model.wv[token]
            self.voc.binary_vocabulary[token] = vector

    def convert_arrays(self):
        print("Convert arrays into np.array...")
        self.input_images = np.array(self.input_images)
        self.partial_sequences = np.array(self.partial_sequences)
        self.next_words = np.array(self.next_words)

    def append(self, sample_id, gui, img, to_show=False):
        if to_show:
            pic = img * 255
            pic = np.array(pic, dtype=np.uint8)
            Utils.show(pic)

        token_sequence = [START_TOKEN]
        for line in gui:
            line = line.replace(",", " ,").replace("\n", " \n")
            tokens = line.split(" ")
            for token in tokens:
                self.voc.append(token)
                token_sequence.append(token)
        token_sequence.append(END_TOKEN)
        # Insieme dei token che servono per word2vec
        self.data.append(token_sequence)

        suffix = [PLACEHOLDER] * CONTEXT_LENGTH

        a = np.concatenate([suffix, token_sequence])
        for j in range(0, len(a) - CONTEXT_LENGTH):
            context = a[j:j + CONTEXT_LENGTH]
            label = a[j + CONTEXT_LENGTH]
            self.ids.append(sample_id)
            self.input_images.append(img)
            self.partial_sequences.append(context)
            self.next_words.append(label)

    @staticmethod
    def indexify(partial_sequences, voc):
        temp = []
        for sequence in partial_sequences:
            sparse_vectors_sequence = []
            for token in sequence:
                sparse_vectors_sequence.append(voc.vocabulary[token])
            temp.append(np.array(sparse_vectors_sequence))

        return temp

    @staticmethod
    def binarize(partial_sequences, voc):
        temp = []
        for sequence in partial_sequences:
            sparse_vectors_sequence = []
            for token in sequence:
                sparse_vectors_sequence.append(voc.binary_vocabulary[token])
            temp.append(np.array(sparse_vectors_sequence))

        return temp  # (48, 19)

    @staticmethod
    def sparsify_labels(next_words, voc):
        temp = []
        for label in next_words:
            temp.append(voc.binary_vocabulary[label])

        return temp

    def save_metadata(self, path):
        np.save("{}/meta_dataset".format(path), np.array([self.input_shape, self.output_size, self.size]))

    @staticmethod
    def tsne_plot(model):
        "Create TSNE model and plot it"
        labels = []
        tokens = []
    
        for word in model.wv.vocab:
            tokens.append(model[word])
            labels.append(word)
    
        tsne_model = TSNE(perplexity=5, n_components=2, init='pca', n_iter=3500, random_state=23)
        new_values = tsne_model.fit_transform(tokens)

        x = []
        y = []
        for value in new_values:
            x.append(value[0])
            y.append(value[1])
    
        plt.figure(figsize=(18, 18))
        for i in range(len(x)):
            plt.scatter(x[i], y[i])
            plt.annotate(labels[i],
                         xy=(x[i], y[i]),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
        plt.show()

    
    