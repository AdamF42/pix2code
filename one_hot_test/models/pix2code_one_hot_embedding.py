import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout, \
    RepeatVector, concatenate, \
    Conv2D, MaxPooling2D, Flatten

from w2v_test.costants import PLACEHOLDER, START_TOKEN, END_TOKEN, CONTEXT_LENGTH
from w2v_test.dataset.utils import get_preprocessed_img


class Pix2codeOneHotEmbedding(tf.keras.models.Model):

    def __init__(self,
                 embedding_matrix,
                 kernel_shape=(3, 3),
                 activation='relu',
                 context_length=CONTEXT_LENGTH,
                 dropout_ratio=0.25,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        vocab_size = len(embedding_matrix)

        self.image_model_layers = [
            Conv2D(32, kernel_shape, padding='valid', activation=activation),
            Conv2D(32, kernel_shape, padding='valid', activation=activation),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(dropout_ratio),

            Conv2D(64, kernel_shape, padding='valid', activation=activation),
            Conv2D(64, kernel_shape, padding='valid', activation=activation),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(dropout_ratio),

            Conv2D(128, kernel_shape, padding='valid', activation=activation),
            Conv2D(128, kernel_shape, padding='valid', activation=activation),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(dropout_ratio),

            Flatten(),
            Dense(1024, activation='relu'),
            Dropout(0.3),
            Dense(1024, activation='relu'),
            Dropout(0.3),

            RepeatVector(context_length)
        ]

        self.language_model_layers = [
            tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=vocab_size,
                                      name='embedding',
                                      embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                      trainable=False),
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

        for layer in self.image_model_layers:
            img_inp = layer(img_inp)

        for layer in self.language_model_layers:
            context_inp = layer(context_inp)
            if "embedding" in layer.name:
                print(layer.weights)

        decoder = concatenate([img_inp, context_inp])
        for layer in self.decoder_layers:
            decoder = layer(decoder)
        return decoder

    def compile(self, loss='sparse_categorical_crossentropy',
                optimizer=tf.keras.optimizers.RMSprop(lr=0.0002, clipvalue=1.0), metrics=None,
                **kwargs):
        if metrics is None:
            metrics = ['sparse_categorical_accuracy']
        self.output_names = ['code']
        return super().compile(loss=loss, optimizer=optimizer, metrics=metrics, **kwargs)

    @staticmethod
    def predict_image(model, image, vocabulary, max_sentence_len):

        if isinstance(image, str):
            img = np.array([get_preprocessed_img(image)])
        elif isinstance(image, np.ndarray):
            img = image
        else:
            raise TypeError("Unknown handling of image input of type {}".format(type(image)))

        current_context = [vocabulary.inv_token_lookup[PLACEHOLDER]] * (CONTEXT_LENGTH - 1)
        current_context.append(vocabulary.inv_token_lookup[START_TOKEN])

        predictions = [START_TOKEN]

        for i in range(0, max_sentence_len):

            probas = model.predict(x=[img, np.array([current_context])])
            prediction = np.argmax(probas[-1])

            new_context = []
            for j in range(1, CONTEXT_LENGTH):
                new_context.append(current_context[j])

            new_context.append(prediction)

            current_context = new_context

            predictions.append(vocabulary.token_lookup[prediction])

            if vocabulary.token_lookup[prediction] == END_TOKEN:
                break
        return predictions
