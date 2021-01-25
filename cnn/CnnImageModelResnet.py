import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Flatten, Dense, concatenate
from tensorflow.keras.optimizers import RMSprop

from cnn.CnnCounterUnit import CnnCounterUnit
from cnn.CnnModelResnet import CnnUnitResnet
from utils.costants import IMAGE_SIZE, CONTEXT_LENGTH, PLACEHOLDER, END_TOKEN
from utils.utils import get_preprocessed_img
from utils.vocabulary import Vocabulary


class CnnImageModelResnet(tf.keras.Model):

    def __init__(self, words, image_count_words, max_sentence_length=100, kernel_shape=7, dense_layer_size=512, activation='relu',
                 dropout_ratio=0.25, order_layer_output_size=1024,
                 image_out=False, *args, **kwargs):

        super().__init__(*args, **kwargs)
        if isinstance(kernel_shape, tuple) or isinstance(kernel_shape, list):
            kernel_shape = tuple(kernel_shape)
        else:
            kernel_shape = (kernel_shape, kernel_shape)

        self.voc_size = len(words)
        self.image_out = image_out
        self.image_count_words = image_count_words
        self.max_sentence_length = max_sentence_length

        self.shallow_cnn_unit = CnnUnitResnet(image_count_words=image_count_words, kernel_shape=kernel_shape,
                                               dropout_ratio=dropout_ratio, activation=activation, name='cnn_unit')
        self.parallel_counter_unit = CnnCounterUnit(layer_size=dense_layer_size, activation=activation,
                                                 name='counter_unit')

        self.ordering_layers = [
            Flatten(name='ordering_flatten'),
            Dense(1024, activation=activation, name='ordering_1'),
            Dropout(dropout_ratio, name='ordering_drop_1'),
            Dense(1024, activation=activation, name='ordering_2'),
            Dropout(dropout_ratio, name='ordering_drop_2'),
            Dense(order_layer_output_size, activation=activation, name='ordering_3')
        ]

        self.code_output_units = [Dense(self.voc_size, activation='softmax', name="code_out_{}".format(i))
                                  for i in range(max_sentence_length)]

    def call(self, inputs, **kwargs):
        inp = inputs['img_data']
        cnn_output_layers = self.shallow_cnn_unit(inp)
        obj_counter_ouputs = [self.parallel_counter_unit(img_out) for img_out in cnn_output_layers]
        obj_counter_ouputs = {key + "_count": layer for key, layer in zip(self.image_count_words, obj_counter_ouputs)}

        ordering_inp = concatenate(cnn_output_layers)
        for layer in self.ordering_layers:
            ordering_inp = layer(ordering_inp)
        code_ordered_out_layers = [layer(ordering_inp) for layer in self.code_output_units]
        code = concatenate([tf.expand_dims(l, 1) for l in code_ordered_out_layers], axis=1)

        outputs = obj_counter_ouputs
        outputs.update({'code': code})

        if self.image_out:
            outputs.update({"img_out_" + key: layer for key, layer in zip(self.image_count_words,
                                                                          cnn_output_layers)})
        return outputs

    def compile(self, loss='mse', optimizer=RMSprop(lr=0.0001, clipvalue=1.0), *args, **kwargs):
        # Fix the bug in tensorflow 1.15 that sets the outputnames wrong when using dicts and generators
        if self.image_out:
            names = ([key + "_count" for key in self.image_count_words]
                     + ["img_out_" + key for key in self.image_count_words]
                     + ["code"])
        else:
            names = [key + "_count" for key in self.image_count_words] + ['code']
        self.output_names = sorted(names)
        return super().compile(loss=loss, optimizer=optimizer, *args, **kwargs)

    def predict_image(self, image, voc: Vocabulary, img_size=IMAGE_SIZE, context_length=CONTEXT_LENGTH):

        def clean_prediction(pred, voc: Vocabulary):
            pred_code_one_hot = pred['code'][0]

            # voc.index_to_word

            pred_code_tokens = np.argmax(pred_code_one_hot, axis=1)

            predicted_tokens = [voc.index_to_word(i) for i in pred_code_tokens]

            # end_index = np.where(pred_code_tokens == voc.word2token_dict[END_TOKEN])
            # if len(end_index[0]) == 0:
            #     end_index = np.where(pred_code_tokens == voc.word2token_dict[PLACEHOLDER])
            #     if len(end_index[0]) == 0:
            #         end_index = pred_code_tokens.shape[0]
            # end_index = end_index[0][0]
            # pred_code = [voc.token2word_dict[val] for val in (pred_code_tokens[:end_index])]
            # return " ".join(pred_code)
            return predicted_tokens

        if isinstance(image, str):
            img = get_preprocessed_img(image, img_size)
        elif isinstance(image, np.ndarray):
            img = image
        else:
            raise TypeError("Unknown handling of image input of type {}".format(type(image)))

        pred = self.predict({'img_data': img})
        return clean_prediction(pred, voc)
