#!/usr/bin/env python


import os

import gensim
import numpy as np
from gensim.models import Word2Vec

from model.classes.model.pix2code_w2v import pix2code_w2v
from w2v_test.dataset import Dataset

IMG_W2V_TRAIN_DIR = '/home/adamf42/Projects/pix2code/datasets/web/single'
IMG_PATH = '/home/adamf42/Projects/pix2code/datasets/web/single'

print("################################## GENSIM ##################################")

print('\nPreparing the sentences...')

max_sentence_len = 0
sentences = []

for filename in os.listdir(IMG_W2V_TRAIN_DIR):
    if filename.endswith(".gui"):
        with open(os.path.join(IMG_W2V_TRAIN_DIR, filename)) as doc:
            document = []
            document.append("<START>")
            for line in doc.readlines():
                line = line.replace(" ", "  ") \
                    .replace(",", " ,") \
                    .replace("\n", " \n") \
                    .replace("{", " { ") \
                    .replace("}", " } ") \
                    .replace(",", " , ")
                tokens = line.split(" ")
                tokens = map(lambda x: " " if x == "" else x, tokens)
                # tokens = filter(lambda x: False if x == " " else True, tokens)
                # tokens = filter(lambda x: False if x == "\n" else True, tokens)
                for token in tokens:
                    document.append(token)
            document.append("<END>")
            # print(os.path.join(directory, filename))
            # print(document)
            if len(document) > max_sentence_len:
                max_sentence_len = len(document)
            sentences.append(document)

print("MAX SENTENCE LENGHT: " + str(max_sentence_len))
print("NUMBER OF SENTENCIES: " + str(len(sentences)))
print('\nTraining word2vec...')
word_model: Word2Vec = gensim.models.Word2Vec(sentences, size=100, min_count=1, window=3, iter=200)
pretrained_weights = word_model.wv.vectors
vocab_size, emdedding_size = pretrained_weights.shape
print('Result embedding shape:', pretrained_weights.shape)

def word2idx(word):
    return word_model.wv.vocab[word].index

def idx2word(idx):
    return word_model.wv.index2word[idx]


print('\nPreparing the data for LSTM...')
train_x = np.zeros([len(sentences), max_sentence_len], dtype=np.int32)
train_y = np.zeros([len(sentences)], dtype=np.int32)
for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence[:-1]):
        train_x[i, t] = word2idx(word)
    train_y[i] = word2idx(sentence[-1])
print('train_x shape:', train_x.shape)
print('train_y shape:', train_y.shape)

print("emdedding_size: {}, vocab_size: {}".format(emdedding_size, vocab_size))


print("################################## DATASET ##################################")


# generator.__getitem__(0)

dataset = Dataset(word_model)

dataset.load(IMG_PATH)

dataset.create_word2vec_representation()

print(dataset.partial_sequences.shape)

model = pix2code_w2v(input_shape=dataset.input_shape, output_path="ciccio", encoding_type="ciccio",
                     pretrained_weights = pretrained_weights)

model.model.summary()

model.fit(dataset.input_images, dataset.partial_sequences, dataset.next_words)








