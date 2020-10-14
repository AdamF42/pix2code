from __future__ import absolute_import

__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

from keras import *
from keras.layers import Dense, RepeatVector, LSTM, concatenate, Flatten, Dropout
from tensorflow.python.keras.applications.resnet_v2 import ResNet50V2

from .AModel import *
from .Config import *


class pix2codeResNet(AModel):
    name: str = "resnet"

    def __init__(self, input_shape, output_size, output_path, encoding_type):
        AModel.__init__(self, input_shape, output_size, output_path, encoding_type)

        image_model = Sequential()
        image_model.add(ResNet50V2(include_top=False, weights=None, input_shape=input_shape))

        image_model.add(Flatten())
        image_model.add(Dense(1024, activation='relu'))
        image_model.add(Dropout(0.3))
        image_model.add(Dense(1024, activation='relu'))
        image_model.add(Dropout(0.3))

        image_model.add(RepeatVector(CONTEXT_LENGTH))

        visual_input = Input(shape=input_shape)
        encoded_image = image_model(visual_input)

        language_model = Sequential()
        language_model.add(LSTM(128, return_sequences=True, input_shape=(CONTEXT_LENGTH, output_size)))
        language_model.add(LSTM(128, return_sequences=True))

        textual_input = Input(shape=(CONTEXT_LENGTH, output_size))
        encoded_text = language_model(textual_input)

        # should be(None, 48, 1024) (None, 48, 128)
        decoder = concatenate([encoded_image, encoded_text])  # (None, 8, 8, 2048) (None, 48, 128)

        decoder = LSTM(512, return_sequences=True)(decoder)
        decoder = LSTM(512, return_sequences=False)(decoder)
        decoder = Dense(output_size, activation='softmax')(decoder)

        self.model = Model(inputs=[visual_input, textual_input], outputs=decoder)
        self.compile()

    def fit(self, images, partial_captions, next_words):
        self.model.fit([images, partial_captions], next_words, shuffle=False, epochs=EPOCHS,
                       batch_size=BATCH_SIZE, verbose=1)
        self.save()

    def fit_generator(self, generator, steps_per_epoch, epochs=EPOCHS):
        self.model.fit_generator(generator, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=1)
        self.save()

    def predict_batch(self, images, partial_captions):
        return self.model.predict([images, partial_captions], verbose=1)

    def predict(self, image, partial_caption):
        return self.model.predict([image, partial_caption], verbose=0)[0]
