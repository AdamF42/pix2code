from __future__ import absolute_import

import tensorflow as tf
from keras.layers import Dense, Dropout, \
    Conv2D, MaxPooling2D, Flatten


class Pix2CodeOriginalCnnModel(tf.keras.models.Model):

    def __init__(self,
                 output_names,
                 kernel_shape=(3, 3),
                 activation='relu',
                 dropout_ratio=0.25,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.layer_output_names = output_names
        self.output_length = len(output_names)
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
            [Dense(1, name=output_names[i], activation=activation)
             for i in range(self.output_length)]
            # RepeatVector(context_length)
        ]

    def call(self, inputs, **kwargs):

        for layer in self.image_model_layers[:-1]:
            inputs = layer(inputs)

        last_layers = self.image_model_layers[-1]
        out = {name + "_count": layer(inputs)
               for name, layer in zip(self.layer_output_names, last_layers)}

        return out

    def compile(self, loss='mse',
                optimizer=tf.keras.optimizers.RMSprop(lr=0.0002, clipvalue=1.0), metrics=None,
                **kwargs):
        if metrics is None:
            metrics = ['accuracy']
        self.output_names = sorted([key + "_count" for key in self.layer_output_names])
        return super().compile(loss=loss, optimizer=optimizer, metrics=metrics, **kwargs)

    def predict(self, *args, return_as_dict=True, **kwargs):
        pred = super().predict(*args, **kwargs)
        if return_as_dict:
            return {key: val for key, val in zip(self.layer_output_names, pred)}
        else:
            return pred
