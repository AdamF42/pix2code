from __future__ import absolute_import

import tensorflow as tf
from keras.layers import Dense, Dropout, \
    RepeatVector, concatenate, \
    Conv2D, MaxPooling2D, Flatten
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop

CONTEXT_LENGTH = 48


class Pix2codeW2VEmbedding(tf.keras.models.Model):

    def __init__(self, pretrained_weights,
                 kernel_shape=(3, 3), activation='relu',
                 context_length=CONTEXT_LENGTH, *args, **kwargs):

        super().__init__(*args, **kwargs)

        vocab_size, emdedding_size = pretrained_weights.shape

        self.image_model_layers = [
            Conv2D(32, kernel_shape, padding='valid', activation=activation),
            Conv2D(32, kernel_shape, padding='valid', activation=activation),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),

            Conv2D(64, kernel_shape, padding='valid', activation=activation),
            Conv2D(64, kernel_shape, padding='valid', activation=activation),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),

            Conv2D(128, kernel_shape, padding='valid', activation=activation),
            Conv2D(128, kernel_shape, padding='valid', activation=activation),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),

            Flatten(),
            Dense(1024, activation='relu'),
            Dropout(0.3),
            Dense(1024, activation='relu'),
            Dropout(0.3),

            RepeatVector(context_length)
        ]

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
        print(img_inp)
        print(context_inp)
        # img_inp = inputs['img_data']
        # context_inp = inputs['context']

        for layer in self.image_model_layers:
            img_inp = layer(img_inp)

        for layer in self.language_model_layers:
            context_inp = layer(context_inp)

        decoder = concatenate([img_inp, context_inp])
        for layer in self.decoder_layers:
            decoder = layer(decoder)
        return {'code': decoder}

    def compile(self, loss='sparse_categorical_crossentropy', optimizer=RMSprop(lr=0.0002, clipvalue=1.0), metrics=None,
                **kwargs):
        if metrics is None:
            metrics = ['categorical_accuracy']
        self.output_names = ['code']
        return super().compile(loss=loss, optimizer=optimizer, metrics=metrics, **kwargs)
