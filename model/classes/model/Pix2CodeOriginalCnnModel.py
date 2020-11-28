import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop


class Pix2CodeOriginalCnnModel(tf.keras.Model):

    def __init__(self, dropout_ratio=0.25, activation='relu', *args, **kwargs):

        super(Pix2CodeOriginalCnnModel, self).__init__(*args, **kwargs)

        self.layer_list = []
        # Input = 256
        self.layer_list.append(
            Conv2D(32, (3, 3), padding='valid', name='cnn_256pix_1', activation=activation, strides=1, dtype='float32'))
        self.layer_list.append(
            Conv2D(32, (3, 3), padding='valid', name='cnn_256pix_2', activation=activation, strides=1))

        self.layer_list.append(MaxPooling2D(pool_size=(2, 2)))
        self.layer_list.append(Dropout(dropout_ratio))
        # 128
        self.layer_list.append(
            Conv2D(64, (3, 3), padding='valid', name='cnn_128pix_1', activation=activation, strides=1))
        self.layer_list.append(
            Conv2D(64, (3, 3), padding='valid', name='cnn_128pix_2', activation=activation, strides=1))

        self.layer_list.append(MaxPooling2D(pool_size=(2, 2)))
        self.layer_list.append(Dropout(dropout_ratio))
        # 64
        self.layer_list.append(Conv2D(128, (3, 3), padding='valid', name='cnn_64pix_1', activation=activation))
        self.layer_list.append(Conv2D(128, (3, 3), padding='valid', name='cnn_64pix_2', activation=activation))
        self.layer_list.append(MaxPooling2D(pool_size=(2, 2)))
        self.layer_list.append(Dropout(dropout_ratio))

        self.layer_list.append(Flatten())
        self.layer_list.append(Dense(1024, activation=activation))
        self.layer_list.append(Dropout(0.3))
        self.layer_list.append(Dense(1024, activation=activation))
        self.layer_list.append(Dropout(0.3))
        self.layer_list.append(Dense(19, activation="softmax"))

    def call(self, inputs, **kwargs):
        print("shape: ", tf.shape(inputs))
        inp = inputs
        for layer in self.layer_list:
            # print(layer.name + " shape: ", tf.shape(inp))
            inp = layer(inp)  # inp = tensor (1, 256, 256, 3) dtype=float32

        out = {self.layer_list[-1].name: self.layer_list[-1]}
        return out

    def compile(self, loss='mse', optimizer=RMSprop(lr=0.0001, clipvalue=1.0), **kwargs):
        self.output_names = self.layer_list[-1].name
        return super().compile(loss=loss, optimizer=optimizer, **kwargs)

    def predict(self, *args, return_as_dict=True, **kwargs):
        pred = super().predict(*args, **kwargs)
        if return_as_dict:
            return {key: val for key, val in zip(self.output_names, pred)}
        else:
            return pred

    def get_config(self):
        pass
