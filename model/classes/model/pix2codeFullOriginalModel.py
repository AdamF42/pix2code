import tensorflow as tf

from tensorflow.python.keras.layers import Input, Dense, Dropout, \
                         RepeatVector, concatenate, \
                         Conv2D, MaxPooling2D, Flatten

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Sequential

CONTEXT_LENGTH = 48


class pix2codeFullOriginalModel(tf.keras.models.Model):

    def __init__(self, input_shape, output_size, *args, **kwargs):

        super().__init__(*args, **kwargs)

        image_model = tf.keras.Sequential()
        image_model.add(Conv2D(32, (3, 3), padding='valid', name='cnn_256pix_1', activation='relu', input_shape=input_shape))
        image_model.add(Conv2D(32, (3, 3), padding='valid', name='cnn_256pix_2', activation='relu'))
        image_model.add(MaxPooling2D(pool_size=(2, 2)))
        image_model.add(Dropout(0.25))

        image_model.add(Conv2D(64, (3, 3), padding='valid', name='cnn_128pix_1', activation='relu'))
        image_model.add(Conv2D(64, (3, 3), padding='valid', name='cnn_128pix_2', activation='relu'))
        image_model.add(MaxPooling2D(pool_size=(2, 2)))
        image_model.add(Dropout(0.25))

        image_model.add(Conv2D(128, (3, 3), padding='valid', name='cnn_64pix_1', activation='relu'))
        image_model.add(Conv2D(128, (3, 3), padding='valid', name='cnn_64pix_2', activation='relu'))
        image_model.add(MaxPooling2D(pool_size=(2, 2)))
        image_model.add(Dropout(0.25))

        image_model.add(Flatten())
        image_model.add(Dense(1024, activation='relu'))
        image_model.add(Dropout(0.3))
        image_model.add(Dense(1024, activation='relu'))
        image_model.add(Dropout(0.3))

        image_model.add(RepeatVector(CONTEXT_LENGTH))

        visual_input = Input(shape=input_shape)
        encoded_image = image_model(visual_input)

        language_model = Sequential()
        language_model.add(tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(CONTEXT_LENGTH, output_size)))
        language_model.add(tf.keras.layers.LSTM(128, return_sequences=True))

        textual_input = Input(shape=(CONTEXT_LENGTH, output_size))
        encoded_text = language_model(textual_input)

        decoder = concatenate([encoded_image, encoded_text])

        decoder = tf.keras.layers.LSTM(512, return_sequences=True)(decoder)
        decoder = tf.keras.layers.LSTM(512, return_sequences=False)(decoder)
        decoder = Dense(output_size, activation='softmax')(decoder)

        super().__init__(inputs=[visual_input, textual_input], outputs=decoder)

    def call(self, inputs, **kwargs):
        img_inp, context_inp = inputs

        for layer in self.image_model_layers:
            img_inp = layer(img_inp)

        for layer in self.language_model_layers:
            context_inp = layer(context_inp)

        decoder = concatenate([img_inp, context_inp])
        for layer in self.decoder_layers:
            decoder = layer(decoder)
        return {'code': decoder}

    def compile(self, loss='sparse_categorical_crossentropy', optimizer=RMSprop(lr=0.0001, clipvalue=1.0), metrics=None,
                **kwargs):
        if metrics is None:
            metrics = ['categorical_accuracy']
        self.output_names = ['code']
        return super().compile(loss=loss, optimizer=optimizer, metrics=metrics, **kwargs)

    def predict(self, *args, return_as_dict=True, **kwargs):
        pred = super().predict(*args, **kwargs)
        if return_as_dict:
            return {key: val for key, val in zip(self.output_names, pred)}
        else:
            return pred
