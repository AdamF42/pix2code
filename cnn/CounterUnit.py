import tensorflow as tf
from keras.layers import Dense, Flatten


class CounterUnit(tf.keras.layers.Layer):
    def __init__(self, layer_size, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.main_layers = [
            Flatten(name='flatten'),
            Dense(layer_size, activation=activation, name="counter1"),
            Dense(layer_size, activation=activation, name="counter2")]
        self.last_layer = Dense(1, activation=activation, name='counter_out')

    def call(self, inputs, **kwargs):
        inp = inputs
        for layer in self.main_layers:
            inp = layer(inp)
        return self.last_layer(inp)
