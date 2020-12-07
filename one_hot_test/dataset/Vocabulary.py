import numpy as np
from w2v_test.costants import *


class Vocabulary:
    def __init__(self):
        self.binary_vocabulary = {}
        self.vocabulary = {}  # {'<START>': 0, '<END>': 1, ' ': 2, 'header': 3, . . . }
        self.token_lookup = {}  # {0: '<START>', 1: '<END>', 2: ' ', 3: 'header', . . . }
        self.inv_token_lookup = {}  # {'<START>': 0, '<END>': 1, ' ': 2, 'header': 3, . . . }
        self.embedding_matrix = None
        self.size = 0

        self.append(START_TOKEN)
        self.append(END_TOKEN)
        self.append(PLACEHOLDER)

    def append(self, token):
        if token not in self.vocabulary:
            self.vocabulary[token] = self.size
            self.token_lookup[self.size] = token
            self.inv_token_lookup[token] = self.size
            self.size += 1

    def create_binary_voc(self):
        for token in self.vocabulary:
            binary = np.zeros(self.size)
            binary[self.vocabulary[token]] = 1
            self.binary_vocabulary[token] = binary

    def create_embedding_matrix(self):
        embedding_dim = self.size
        num_tokens = self.size
        self.embedding_matrix = np.zeros((num_tokens, embedding_dim))
        for i in range(num_tokens):
            label = self.token_lookup[i]
            embedding_vector = self.binary_vocabulary[label]
            if embedding_vector is not None:
                self.embedding_matrix[i] = embedding_vector
