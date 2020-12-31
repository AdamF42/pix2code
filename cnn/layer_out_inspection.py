import cv2
import matplotlib.pyplot as plt
import numpy as np

from cnn.CnnModel import CnnModel
from cnn.generator import DataGenerator
from utils.utils import load_pickle
from w2v_test.costants import TOKEN_TO_EXCLUDE, COMMA, START_TOKEN, END_TOKEN
from w2v_test.dataset.dataset import Dataset


def inspect_layer(img_data, img_name):
    pred_all = new_model.predict(img)
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


output_names = load_pickle('/home/adamf42/Projects/pix2code/pickle/output_names.pickle')

IMG_PATH = '/home/adamf42/Projects/pix2code/datasets/web/prove'

tokens_to_exclude = TOKEN_TO_EXCLUDE + [COMMA, START_TOKEN, END_TOKEN]
labels, img_paths = Dataset.load_paths_only(IMG_PATH)
generator = DataGenerator(img_paths, labels, output_names, tokens_to_exclude, batch_size=1)

new_model = CnnModel(output_names, image_out=True, dense_layer_size=512, dropout_ratio=0.1)
new_model.compile()
img_build, _ = generator.__getitem__(0)
new_model.predict(img_build)

new_model.load_weights('best_model_cnn_final.h5')

data = [generator.__getitem__(i) + ({'name': img_paths[i]},) for i in range(len(img_paths))]

for img, _, name in data:
    inspect_layer(img['img_data'], name['name'])
