import os

import numpy as np

from w2v_test.costants import IMAGE_SIZE, START_TOKEN, END_TOKEN, TOKEN_TO_EXCLUDE, PLACEHOLDER, CARRIAGE_RETURN


def sparsify(label_vector, output_size):
    sparse_vector = []

    for label in label_vector:
        sparse_label = np.zeros(output_size)
        sparse_label[label] = 1

        sparse_vector.append(sparse_label)

    return np.array(sparse_vector)


def get_preprocessed_img(img_path, image_size=IMAGE_SIZE):
    import cv2
    img = cv2.imread(img_path)
    img = cv2.resize(img, (image_size, image_size))
    img = img.astype('float32')
    img /= 255
    return img


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
        tokens = map(lambda x: " " if x == "" else x, tokens)
        tokens = map(lambda x: PLACEHOLDER if x == " " else x, tokens)
        tokens = map(lambda x: CARRIAGE_RETURN if x == "\n" else x, tokens)
        tokens = filter(lambda x: False if x in tokens_to_exclude else True, tokens)
        for token in tokens:
            token_sequence.append(token)
    token_sequence.append(END_TOKEN)
    return token_sequence
