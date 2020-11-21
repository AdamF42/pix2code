from __future__ import absolute_import

__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

from keras import *
from keras.layers import Dense, Dropout, \
    RepeatVector, LSTM, concatenate, \
    Flatten
from tensorflow.python.keras.applications.vgg16 import VGG16

from .AModel import *
from .Config import *


class pix2codeVGG16(AModel):
    name = "VGG16"

    def __init__(self, input_shape, output_size, output_path, encoding_type):
        self.name = self.name

    AModel.__init__(self, input_shape, output_size, output_path, encoding_type)

    image_model = Sequential()
    image_model.add(VGG16(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=input_shape,
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
    ))

    image_model.add(Flatten())
    image_model.add(Dense(4096, activation='relu'))
    image_model.add(Dropout(0.5))
    image_model.add(Dense(4096, activation='relu'))
    image_model.add(Dropout(0.5))
    image_model.add(Dense(1000, activation='softmax'))

    image_model.add(RepeatVector(CONTEXT_LENGTH))

    visual_input = Input(shape=input_shape)
    encoded_image = image_model(visual_input)

    language_model = Sequential()
    language_model.add(LSTM(128, return_sequences=True, input_shape=(CONTEXT_LENGTH, output_size)))
    language_model.add(LSTM(128, return_sequences=True))

    textual_input = Input(shape=(CONTEXT_LENGTH, output_size))
    encoded_text = language_model(textual_input)

    decoder = concatenate([encoded_image, encoded_text])

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
        self.model.fit(generator, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=2, batch_size=BATCH_SIZE)

    self.save()

    def predict_batch(self, images, partial_captions):
        return self.model.predict([images, partial_captions], verbose=1)

    def predict(self, image, partial_caption):
        return self.model.predict([image, partial_caption], verbose=0)[0]
