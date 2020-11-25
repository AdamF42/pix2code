#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os

import gensim
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import LambdaCallback
from keras.layers import Dense, Input
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.manifold import TSNE

IMG_DIR = './datasets/web/single'

def tsne_plot(model):
    "Create TSNE model and plot it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model.wv.__getitem__(word))
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


print('\nPreparing the sentences...')

max_sentence_len = 0
sentences = []

for filename in os.listdir(IMG_DIR):
    if filename.endswith(".gui"):
        with open(os.path.join(IMG_DIR, filename)) as doc:
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
                tokens = filter(lambda x: False if x == " " else True, tokens)
                tokens = filter(lambda x: False if x == "\n" else True, tokens)
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
word_model = gensim.models.Word2Vec(sentences, size=100, min_count=1, window=3, iter=200)
pretrained_weights = word_model.wv.vectors
vocab_size, emdedding_size = pretrained_weights.shape
print('Result embedding shape:', pretrained_weights.shape)

# print("VOCABULARY")
# for word in word_model.wv.vocab.keys():
#     print(word+"\n")

# tsne_plot(word_model)


# print('Checking similar words:')
# for word in word_model.wv.vocab.keys():
#     most_similar = ', '.join('%s (%.2f)' % (similar, dist) for similar, dist in word_model.wv.most_similar(word)[:8])
#     print('  %s -> %s' % (word, most_similar))
#


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

print("emdedding_size: {}, vocab_size: {}".format(emdedding_size,vocab_size))

print('\nTraining LSTM...')
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, weights=[pretrained_weights]))

model.add(LSTM(units=emdedding_size, return_sequences=True))
model.add(LSTM(units=emdedding_size))

model.add(Dense(units=vocab_size, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

input = Input(name='input_lan_model')
encoded_text = model(input)

model.summary()


def sample(preds, temperature=1.0):
    if temperature <= 0:
        return np.argmax(preds)
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_next(text, num_generated=10):
    word_idxs = [word2idx(word) for word in text.lower().split()]
    for i in range(num_generated):
        prediction = model.predict(x=np.array(word_idxs))
        idx = sample(prediction[-1], temperature=0.7)
        # idx = np.argmax(prediction)
        word_idxs.append(idx)
    return ' '.join(idx2word(idx) for idx in word_idxs)


def on_epoch_end(epoch, _):
    print('\nGenerating text after epoch: %d' % epoch)
    texts = [
        'header',
        '{',
        'row',
        ',',
        'quadruple',
    ]
    for text in texts:
        sample = generate_next(text)
        print('%s... -> %s' % (text, sample))


def on_epoch_begin(epoch,_):
    print('\nepoch: %d' % epoch)



# model.fit(train_x, train_y,
#           batch_size=64,
#           epochs=5,
#           callbacks=[LambdaCallback(on_epoch_end=on_epoch_end), LambdaCallback(on_epoch_begin=on_epoch_begin)])

print(train_x.shape)

# model.fit(train_x, train_y,
#           batch_size=64,
#           epochs=5)