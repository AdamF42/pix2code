import math
from functools import reduce

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.models import load_model

from original_cnn.Pix2CodeOriginalCnnModel import Pix2CodeOriginalCnnModel
from w2v_test.costants import TOKEN_TO_EXCLUDE, COMMA, START_TOKEN, END_TOKEN, CNN_OUTPUT_NAMES
from w2v_test.dataset.utils import get_token_sequences_with_max_seq_len

IMG_PATH = "/home/filippo/Project/pix2code/datasets/web/single"

print("################################## MODEL ##################################")

tokens_to_exclude = TOKEN_TO_EXCLUDE + [COMMA, START_TOKEN, END_TOKEN]

tokens = get_token_sequences_with_max_seq_len(IMG_PATH, tokens_to_exclude)

voc = {token for token in reduce(lambda x, y: x + y, tokens['sentences'])}
output_names = list(map(lambda x: CNN_OUTPUT_NAMES[x] if x in CNN_OUTPUT_NAMES.keys() else x, voc))

model_instance = Pix2CodeOriginalCnnModel(output_names)
# Build the model
model_instance.compile()
model_instance: Pix2CodeOriginalCnnModel = load_model('../pix2code_cnn_china_mse_15patient')

id_img = "0B660875-60B4-4E65-9793-3C7EB6C8AFD0"
npz_file = id_img + ".npz"

temp = np.load(IMG_PATH + '/' + npz_file, allow_pickle=True)
img_data = temp['features']
pred_all = model_instance.predict({'img_data': np.reshape(img_data, tuple([1] + list(img_data.shape)))})
pred_all = {key: val[0] for key, val in pred_all.items()}
n_plots = int(len(pred_all) / 2) + 1
n_rows = math.ceil(n_plots / 3)
fig, ax = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows), squeeze=False)

ax[0, 0].imshow(cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB), )
for i, word in enumerate(voc, start=1):
    if n_rows > 1:
        ax[int(i / 3), i % 3].imshow(pred_all['{}_count'.format(word)].squeeze(-1))
        ax[int(i / 3), i % 3].set_title("{}:{}".format(word, pred_all[word + "_count"][0]))
# plt.savefig('image_out_example.png')
