#!/usr/bin/env python
from __future__ import print_function
from __future__ import absolute_import

import sys

import numpy as np
import tensorflow as tf
from keras.models import Model
from matplotlib import pyplot as plt

from model.classes.model.pix2codeFullOriginalModel import pix2codeFullOriginalModel
from model.classes.model.Pix2CodeOriginalCnnModel import Pix2CodeOriginalCnnModel

from classes.dataset.Generator import *


def main(input_path, output_path, encoding_type, is_memory_intensive=False):

    np.random.seed(42)

    dataset = Dataset()
    if encoding_type == "one_hot":
        dataset.load_with_one_hot_encoding(input_path, generate_binary_sequences=True)
    elif encoding_type == "w2v":
        dataset.load_with_word2vec(input_path, generate_binary_sequences=True)
    else:
        raise Exception("Missing parameter")
    dataset.save_metadata(output_path)
    dataset.voc.save(output_path)

    if not is_memory_intensive:
        dataset.convert_arrays()

        input_shape = dataset.input_shape
        output_size = dataset.output_size

        print(len(dataset.input_images), len(dataset.partial_sequences), len(dataset.next_words))
        print(dataset.input_images.shape, dataset.partial_sequences.shape, dataset.next_words.shape)
    else:

        input_shape = dataset.input_shape
        output_size = dataset.output_size

        voc = Vocabulary()
        voc.retrieve(output_path)

    layer_name = "sequential"
    img_path = '../../pix2code/datasets/web/eval_set/0D99F46A-BEDB-444C-B948-246096DFEBD4.png'
    current_context = [voc.vocabulary[PLACEHOLDER]] * CONTEXT_LENGTH
    evaluation_img = Utils.get_preprocessed_img(img_path, IMAGE_SIZE)
    evaluation_img = np.array([evaluation_img])

    model = pix2codeFullOriginalModel(input_shape=input_shape, output_size=output_size)
    # model = Pix2CodeOriginalCnnModel()

    if type(model) == Pix2CodeOriginalCnnModel:
        # Use list of layer
        print(evaluation_img.shape)
        print(tf.shape(evaluation_img))
        model.build(evaluation_img.shape)  # model(evaluation_img)
        # model.compile()
        model.summary()

        layer_to_print = []
        filter_to_print = []
        for layer in model.layer_list:
            if "cnn" in layer.name:
                layer_to_print.append(layer)
                # get filters
                filters, biases = layer.get_weights()
                # print(filters.shape)
                # normalize filter values to 0-1 so we can visualize them
                f_min, f_max = filters.min(), filters.max()
                filters = (filters - f_min) / (f_max - f_min)
                filter_to_print.append(filters)

        print_filters(filter_to_print)
        # print(filter_to_print)

        input_model = tf.keras.Input(shape=evaluation_img.shape)  # shape=(None, 1, 256, 256, 3), dtype=float32)
        for el in layer_to_print:
            print("el: ", el)
            print("oltro: ", el.name)
            # print("oltro: ", input.shape)
            # inputs should be Tensor("cnn_256pix_1_input:0", shape=(None, 256, 256, 3), dtype=float32)
            intermediate_layer_model = Model(inputs=input_model, outputs=el(input_model))
            intermediate_output = intermediate_layer_model(evaluation_img)
            plt.matshow(intermediate_output[0, :, :, intermediate_output.shape[3]-1], cmap='viridis')
            plt.show()
            print("input: ", intermediate_layer_model.inputs)
            print("output: ", el.output.name)
            print("shape: ", intermediate_layer_model.output.shape)
            input_model = tf.keras.Input(shape=intermediate_layer_model.output.shape)
            print("DONE")


    elif type(model) == pix2codeFullOriginalModel:
        # Use Sequential()
        print_model_info(model, "model.png")

        print("shape immagine: ", evaluation_img.shape)  # 1, 256, 256, 3
        print("shape context: ", np.array(current_context).shape)  # 48, 19
        print("input layer: ", model.input)  # None, 256, 256, 3   -  None, 48, 19
        seq = model.get_layer(layer_name)

        layer_to_print = []
        filter_to_print = []
        for layer in seq.layers:
            if "cnn" in layer.name:
                layer_to_print.append(layer)
                # get filters
                filters, biases = layer.get_weights()
                # print(filters.shape)
                # normalize filter values to 0-1 so we can visualize them
                f_min, f_max = filters.min(), filters.max()
                filters = (filters - f_min) / (f_max - f_min)
                filter_to_print.append(filters)

        # print(layer_to_print)

        print_filters(filter_to_print)

        for el in layer_to_print:
            intermediate_layer_model = Model(inputs=model.get_layer(layer_name).input
                                             , outputs=el.output)
            print("input: ", intermediate_layer_model.inputs)
            print("output: ", el.output.name)
            print("shape: ", intermediate_layer_model.output.shape)
            intermediate_output = intermediate_layer_model.predict(evaluation_img)

            # intermediate_output = intermediate_layer_model(evaluation_img)
            # print("shape: ", intermediate_output.shape)
            plt.matshow(intermediate_output[0, :, :, intermediate_output.shape[3]-1], cmap='viridis')
            plt.show()
            print("DONE")

    else:
        print("ERROR")


def print_model_info(model, file):
    tf.keras.utils.plot_model(
        model,
        to_file=file,
        show_shapes=True
    )
    print(full_summary(model))


def full_summary(layer):
    # check if this layer has layers
    if hasattr(layer, 'layers'):
        print('Summary for ' + layer.name)
        layer.summary()
        print('\n\n')

        for l in layer.layers:
            full_summary(l)


def print_filters(filter_to_print):
    n_filters, ix = 6, 1
    for i in range(n_filters):
        # get the filter
        f = filter_to_print[i][:, :, :, i]
        # plot each channel separately
        for j in range(3):
            # specify subplot and turn of axis
            ax = plt.subplot(n_filters, 3, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(f[:, :, j], cmap='gray')
            ix += 1
    # show the figure
    plt.show()


if __name__ == "__main__":
    argv = sys.argv[1:]

    # example: plot_layer.py ../datasets/web/training_features ../bin one_hot 1

    if len(argv) < 3:
        print("Error: not enough argument supplied:")
        print(
            "plot_layer.py <input path> <output path> <vocabulary encoding type> <is memory intensive (default: 0)> <pretrained weights (optional)> ")
        exit(0)
    else:
        input_path = argv[0]
        output_path = argv[1]
        encoding_type = argv[2]
        use_generator = False if len(argv) < 4 else True if int(argv[3]) == 1 else False

    main(input_path, output_path, encoding_type, is_memory_intensive=use_generator)
