import os
import pickle
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
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


def get_token_sequences_with_max_seq_len(img_dir, tokens_to_exclude=TOKEN_TO_EXCLUDE, is_with_output_name=False):
    max_sentence_len = 0
    sequences = []
    for filename in os.listdir(img_dir):
        if not filename.endswith(".gui"):
            continue
        gui = open(f'{img_dir}/{filename}', 'r')
        token_sequences = get_token_from_gui(gui, tokens_to_exclude)
        if is_with_output_name:
            token_sequences = get_output_names(token_sequences)
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


def save_pickle(obj, pickle_path):
    file_to_store = open(pickle_path, 'wb')
    pickle.dump(obj, file_to_store)
    file_to_store.close()
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


def inspect_layer(model_instance, img, name, output_names):
    img_name = name['name']
    img_data = img['img_data']
    pred_all = model_instance.predict(img)
    pred_all = {key: val[0] for key, val in pred_all.items()}
    n_plots = int(len(pred_all) / 2) + 1
    n_rows = np.math.ceil(n_plots / 3)
    fig, ax = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows), squeeze=False)

    # TODO: fix color
    # print(img_data.shape)
    # test = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
    # print(test.shape)
    test1 = cv2.imread(img_name)
    test1 = cv2.resize(test1, (256, 256))

    ax[0, 0].imshow(test1)
    ax[0, 0].set_title(img_name.split('/')[-1])
    for i, word in enumerate(output_names, start=1):
        if n_rows > 1:
            ax[int(i / 3), i % 3].imshow(pred_all['img_out_{}'.format(word)].squeeze(-1))
            ax[int(i / 3), i % 3].set_title("{}:{}".format(word, pred_all[word + "_count"][0]))
    plt.savefig(img_name.split('/')[-1])
    plt.show()


def plot_cnn_history(history):
    out = list(filter(lambda x: False if x.startswith('val_') else True, history.history.keys()))
    for name in out:
        plt.plot(history.history[name])
        plt.plot(history.history['val_' + name])
        plt.title(name)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


def tsne_plot(model, path):
    "Create TSNE model and plot it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model.wv.__getitem__(word))
        labels.append(word)

    tsne_model = TSNE(perplexity=5, n_components=2, init='pca', n_iter=3500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(18, 18))
    plt.title(path.split('/')[-1])
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(path)
    plt.show()
