from __future__ import absolute_import

import tensorflow as tf
from keras.layers import Dense, Dropout, \
    Conv2D, MaxPooling2D, Flatten


class CnnModelOriginal(tf.keras.models.Model):

    def get_config(self):
        pass

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
        ]

    def call(self, inputs, **kwargs):
        inp = inputs['img_data']
        for layer in self.image_model_layers[:-1]:
            inp = layer(inp)
        last_layers = self.image_model_layers[-1]
        out = {name + "_count": layer(inp) for name, layer in zip(self.layer_output_names, last_layers)}
        return out

    def compile(self, loss='mse',
                optimizer=tf.keras.optimizers.RMSprop(lr=0.0002, clipvalue=1.0), metrics=None,
                **kwargs):
        if metrics is None:
            metrics = ['MeanSquaredError']
        self.output_names = sorted([key + "_count" for key in self.layer_output_names])
        return super().compile(loss=loss, optimizer=optimizer, metrics=metrics, **kwargs)
