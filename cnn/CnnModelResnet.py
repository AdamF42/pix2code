import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout
from tensorflow.keras.optimizers import RMSprop

from cnn.CnnCounterUnit import CnnCounterUnit


class CnnUnitResnet(tf.keras.layers.Layer):
    def __init__(self, image_count_words, kernel_shape=7, dropout_ratio=0.1, activation='relu', **kwargs):
        super().__init__(**kwargs)

        if isinstance(kernel_shape, tuple) or isinstance(kernel_shape, list):
            kernel_shape = tuple(kernel_shape)
        else:
            kernel_shape = (kernel_shape, kernel_shape)

        self.core_cnn_layer_resnet = tf.keras.applications.ResNet152V2(
            include_top=False,
            classes=len(image_count_words),
            classifier_activation="softmax",
        )
        # the parallel cnn layers
        self.parallel_cnn_output_layers = [Conv2D(1, kernel_shape, padding='same', name=layer_name)
                                           for layer_name in image_count_words]

    def call(self, inputs, **kwargs):
        inp = inputs
        inp = self.core_cnn_layer_resnet(inp)
        out_layers = [layer(inp) for layer in self.parallel_cnn_output_layers]
        return out_layers


class CnnModelResnet(tf.keras.Model):

    def get_config(self):
        pass

    def __init__(self, image_count_words, kernel_shape=7, dense_layer_size=256, activation='relu', dropout_ratio=0.25,
                 image_out=False, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.output_length = len(image_count_words)
        self.image_out = image_out
        self.image_count_words = image_count_words
        self.shallow_cnn_unit = CnnUnitResnet(image_count_words=image_count_words, kernel_shape=kernel_shape,
                                        dropout_ratio=dropout_ratio, activation=activation, name='cnn_unit')
        self.parallel_counter_unit = CnnCounterUnit(layer_size=dense_layer_size, activation=activation,
                                                    name='counter_unit')

    def call(self, inputs, **kwargs):
        inp = inputs['img_data']
        cnn_output_layers = self.shallow_cnn_unit(inp)
        output_layers = [self.parallel_counter_unit(img_out) for img_out in cnn_output_layers]
        output_layers = {key + "_count": layer for key, layer in zip(self.image_count_words, output_layers)}
        if self.image_out:
            output_layers.update({"img_out_" + key: layer for key, layer in zip(self.image_count_words,
                                                                                cnn_output_layers)})
        return output_layers

    def compile(self, loss='mse', optimizer=RMSprop(lr=0.0001, clipvalue=1.0), *args, **kwargs):
        if self.image_out:
            names = ([key + "_count" for key in self.image_count_words]
                     + ["img_out_" + key for key in self.image_count_words])
        else:
            names = [key + "_count" for key in self.image_count_words]
        self.output_names = sorted(names)
        return super().compile(loss=loss, optimizer=optimizer, *args, **kwargs)

    def predict(self, *args, return_as_dict=False, **kwargs):
        pred = super().predict(*args, **kwargs)
        if return_as_dict:
            test = {key: val for key, val in pred}
            return test
        else:
            return pred
