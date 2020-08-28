#!/usr/bin/env python
from __future__ import print_function
from __future__ import absolute_import

__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

import numpy as np
import tensorflow.compat.v1 as tf
from classes.model.factory import ModelFactory

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

from classes.dataset.Generator import *
from classes.model.pix2codebilstm import *


def run(input_path, output_path, model_type, encoding_type, is_memory_intensive=False, pretrained_model=None):
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
        gui_paths, img_paths = Dataset.load_paths_only(input_path)

        input_shape = dataset.input_shape
        output_size = dataset.output_size
        steps_per_epoch = dataset.size / BATCH_SIZE

        voc = Vocabulary()
        voc.retrieve(output_path)

        generator = get_generator(dataset, encoding_type, gui_paths, img_paths, voc)

    model = ModelFactory.create_model(model_type, input_shape, output_size, output_path)
    model.model.summary()
    #    tf.keras.utils.plot_model(model, to_file="dot_img_file.png", show_shapes=True)

    if pretrained_model is not None:
        model.model.load_weights(pretrained_model)

    if not is_memory_intensive:
        model.fit(dataset.input_images, dataset.partial_sequences, dataset.next_words)
    else:
        model.fit_generator(generator, steps_per_epoch=steps_per_epoch)

    model.get_time_history()


def get_generator(dataset, encoding_type, gui_paths, img_paths, voc):
    if encoding_type == "one_hot":
        return Generator.data_generator_one_hot(voc, gui_paths, img_paths, batch_size=BATCH_SIZE,
                                                generate_binary_sequences=True)
    elif encoding_type == "w2v":
        return Generator.data_generator_w2v(voc, dataset, gui_paths, img_paths, batch_size=BATCH_SIZE,
                                            generate_binary_sequences=True)
    else:
        raise Exception("Missing parameter")


if __name__ == "__main__":
    argv = sys.argv[1:]

    if len(argv) < 4:
        print("Error: not enough argument supplied:")
        print(
            "train.py <input path> <output path> <model type> <vocabulary encoding type> <is memory intensive (default: 0)> <pretrained weights (optional)> ")
        exit(0)
    else:
        input_path = argv[0]
        output_path = argv[1]
        model_type = argv[2]
        encoding_type = argv[3]
        use_generator = False if len(argv) < 5 else True if int(argv[4]) == 1 else False
        pretrained_weigths = None if len(argv) < 6 else argv[5]

    run(input_path, output_path, model_type, encoding_type, is_memory_intensive=use_generator,
        pretrained_model=pretrained_weigths)
