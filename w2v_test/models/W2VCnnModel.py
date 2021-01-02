from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout, \
    RepeatVector, concatenate, \
    Conv2D, MaxPooling2D, Flatten

from cnn.CnnCounterUnit import CnnCounterUnit
from cnn.CnnModel import CnnUnit
from utils.costants import PLACEHOLDER, START_TOKEN, END_TOKEN, CONTEXT_LENGTH
from utils.utils import get_preprocessed_img
from utils.vocabulary import Vocabulary


class W2VCnnModel(tf.keras.models.Model):

    def get_config(self):
        pass

    def __init__(self,
                 pretrained_weights,
                 image_count_words,
                 dense_layer_size=512,
                 kernel_shape=7,
                 activation='relu',
                 context_length=CONTEXT_LENGTH,
                 dropout_ratio=0.25,
                 image_out=False,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.image_out=image_out
        self.image_count_words=image_count_words
        vocab_size, emdedding_size = pretrained_weights.shape

        self.cnn_unit = CnnUnit(image_count_words=image_count_words, kernel_shape=kernel_shape,
                                               dropout_ratio=dropout_ratio, activation=activation, name='cnn_unit')
        self.counter_unit = CnnCounterUnit(layer_size=dense_layer_size, activation=activation,
                                                 name='counter_unit')
        self.ordering_layers = [
            Flatten(name='ordering_flatten'),
            Dense(1024, activation=activation, name='ordering_1'),
            Dropout(dropout_ratio, name='ordering_drop_1'),
            Dense(1024, activation=activation, name='ordering_2'),
            Dropout(dropout_ratio, name='ordering_drop_2'),
            # TODO: add another ordering layer
        ]

        self.repeat_image_layer = RepeatVector(context_length)

        self.language_model_layers = [
            tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=emdedding_size,
                                      weights=[pretrained_weights], name='embedding'),
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.LSTM(128, return_sequences=True)
        ]

        self.decoder_layers = [
            tf.keras.layers.LSTM(512, return_sequences=True),
            tf.keras.layers.LSTM(512, return_sequences=False),
            Dense(vocab_size, activation='softmax')
        ]

    def call(self, inputs, **kwargs):
        img_inp, context_inp = inputs

        cnn_output_layers = self.cnn_unit(img_inp)
        obj_counter_ouputs = [self.counter_unit(img_out) for img_out in cnn_output_layers]
        obj_counter_ouputs = {key + "_count": layer for key, layer in zip(self.image_count_words, obj_counter_ouputs)}

        ordering_inp = concatenate(cnn_output_layers)
        for layer in self.ordering_layers:
            ordering_inp = layer(ordering_inp)

        img_out = self.repeat_image_layer(ordering_inp)

        for layer in self.language_model_layers:
            context_inp = layer(context_inp)

        decoder = concatenate([img_out, context_inp])
        for layer in self.decoder_layers:
            decoder = layer(decoder)

        outputs = obj_counter_ouputs
        outputs.update({'code': decoder})

        if self.image_out:
            outputs.update({"img_out_" + key: layer for key, layer in zip(self.image_count_words,
                                                                          cnn_output_layers)})
        return outputs

    def compile(self, loss='sparse_categorical_crossentropy',
                optimizer=tf.keras.optimizers.RMSprop(lr=0.0002, clipvalue=1.0), metrics=None,
                **kwargs):
        if metrics is None:
            metrics = ['sparse_categorical_accuracy']

        if self.image_out:
            names = ([key + "_count" for key in self.image_count_words]
                     + ["img_out_" + key for key in self.image_count_words]
                     + ["code"])
        else:
            names = [key + "_count" for key in self.image_count_words] + ['code']
        self.output_names = sorted(names)
        return super().compile(loss=loss, optimizer=optimizer, metrics=metrics, **kwargs)

    def predict_image(self, image, voc: Vocabulary, max_sentence_len):

        def sample(preds, temperature=1.0):
            if temperature <= 0:
                return np.argmax(preds)
            preds = np.asarray(preds).astype('float64')
            preds = np.log(preds) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            probas = np.random.multinomial(1, preds, 1)
            return np.argmax(probas)

        if isinstance(image, str):
            img = np.array([get_preprocessed_img(image)])
        elif isinstance(image, np.ndarray):
            img = image
        else:
            raise TypeError("Unknown handling of image input of type {}".format(type(image)))

        current_context = [voc.word_to_index(PLACEHOLDER)] * (CONTEXT_LENGTH - 1)
        current_context.append(voc.word_to_index(START_TOKEN))

        predictions = [START_TOKEN]

        for i in range(0, max_sentence_len):

            probas = self.predict(x=[img, np.array([current_context])])
            probas = probas['code'][0]
            prediction = sample(probas, temperature=0.7)
            # prediction = np.argmax(probas[-1])

            new_context = []
            for j in range(1, CONTEXT_LENGTH):
                new_context.append(current_context[j])

            new_context.append(prediction)
            current_context = new_context
            predictions.append(voc.index_to_word(prediction))

            if voc.index_to_word(prediction) == END_TOKEN:
                break
        return predictions
