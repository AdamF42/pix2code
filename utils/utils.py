import os
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from w2v_test.costants import IMAGE_SIZE, START_TOKEN, END_TOKEN, TOKEN_TO_EXCLUDE, PLACEHOLDER, CARRIAGE_RETURN, \
    CNN_OUTPUT_NAMES


def sparsify(label_vector, output_size):
    sparse_vector = []

    for label in label_vector:
        sparse_label = np.zeros(output_size)
        sparse_label[label] = 1

        sparse_vector.append(sparse_label)

    return np.array(sparse_vector)


def get_preprocessed_img(img_path, image_size=IMAGE_SIZE):
    if img_path.endswith("png"):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (image_size, image_size))
        img = img.astype('float32')
        img /= 255
        return img
    elif img_path.endswith("npz"):
        return np.load(img_path)["features"]


def show(image):
    import cv2
    cv2.namedWindow("view", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("view", image)
    cv2.waitKey(0)
    cv2.destroyWindow("view")


def get_token_sequences_with_max_seq_len(img_dir, tokens_to_exclude=TOKEN_TO_EXCLUDE):
    max_sentence_len = 0
    sequences = []
    for filename in os.listdir(img_dir):
        if not filename.endswith(".gui"):
            continue
        gui = open(f'{img_dir}/{filename}', 'r')
        token_sequences = get_token_from_gui(gui, tokens_to_exclude)
        sequences.append(token_sequences)
    for sequence in sequences:
        if len(sequence) > max_sentence_len:
            max_sentence_len = len(sequence)
    return {'sentences': sequences, 'max_sentence_len': max_sentence_len}


def get_token_from_gui(gui, tokens_to_exclude=TOKEN_TO_EXCLUDE):
    token_sequence = [START_TOKEN]
    for line in gui:
        line = line.replace(" ", "  ") \
            .replace(",", " ,") \
            .replace("\n", " \n") \
            .replace("{", " { ") \
            .replace("}", " } ") \
            .replace(",", " , ")
        tokens = line.split(" ")
        tokens = list(map(lambda x: " " if x == "" else x, tokens))
        tokens = list(map(lambda x: PLACEHOLDER if x == " " else x, tokens))
        tokens = list(map(lambda x: CARRIAGE_RETURN if x == "\n" else x, tokens))
        token_sequence = token_sequence + tokens
    token_sequence.append(END_TOKEN)
    return list(filter(lambda x: False if x in tokens_to_exclude else True, token_sequence))


def get_output_names(voc_list, out_names_dict=CNN_OUTPUT_NAMES):
    return list(map(lambda x: out_names_dict[x] if x in out_names_dict.keys() else x, voc_list))


def load_pickle(pickle_path):
    file_to_load = open(pickle_path, 'rb')
    obj = pickle.load(file_to_load)
    file_to_load.close()
    return obj


def eval_cnn_model(model_instance, data, words_to_include, index_value='accuracy'):
    prediction_list = []
    correct_y_list = []
    for img, label, name in tqdm(data, desc="Calculating {}".format(index_value)):
        res = model_instance.predict(img)
        correct_y_list.append(pd.DataFrame(label, index=[Path(name['name']).stem]))
        prediction_list.append(pd.DataFrame({word + "_count": res[word + "_count"][0] for word in words_to_include},
                                            index=[Path(name['name']).stem]))

    predictions = pd.concat(prediction_list, axis=0)
    ground_truth = pd.concat(correct_y_list, axis=0)
    ratio_correct_pred = pd.DataFrame((predictions.round() == ground_truth).mean(), columns=[index_value]).T
    return ratio_correct_pred, ground_truth, predictions
