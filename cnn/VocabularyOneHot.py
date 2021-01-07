from utils.vocabulary import Vocabulary
import numpy as np


class VocabularyOneHot(Vocabulary):

    def __init__(self, word_to_encode: dict):
        self.words = list(word_to_encode.keys())
        self.w2encode = word_to_encode
        self.w2i = {val: key for key, val in enumerate(word_to_encode)}
        self.i2w = {val: key for key, val in self.w2i.items()}
        self.binary_vocabulary = {}
        self.embedding_matrix = None
        self.create_binary_voc()
        self.create_embedding_matrix()

    def word_to_index(self, word):
        return self.w2i[word]

    def word_to_encoding(self, word):
        return self.w2encode[word]

    def index_to_word(self, index):
        return  self.i2w[index]

    def get_tokens(self):
        return self.words

    def create_binary_voc(self):
        for token in self.words:
            binary = np.zeros(len(self.words))
            binary[self.w2i[token]] = 1
            self.binary_vocabulary[token] = binary

    def create_embedding_matrix(self):
        embedding_dim = len(self.words)
        num_tokens = len(self.words)
        self.embedding_matrix = np.zeros((num_tokens, embedding_dim))
        for i in range(num_tokens):
            label = self.i2w[i]
            embedding_vector = self.binary_vocabulary[label]
            if embedding_vector is not None:
                self.embedding_matrix[i] = embedding_vector
