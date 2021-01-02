from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.utils.data_utils import iter_sequence_infinite

from CnnModelOriginal import CnnModelOriginal
from generator import DataGenerator
from utils.utils import load_pickle
from utils.costants import TOKEN_TO_EXCLUDE, END_TOKEN, START_TOKEN, COMMA
from utils.dataset import Dataset

IMG_PATH = '../datasets/web/eval_features'
IMG_PATH_EVAL = '../datasets/web/eval_features'

output_names = load_pickle('../pickle/output_names.pickle')

print("################################# GENERATORS ##################################")

tokens_to_exclude = TOKEN_TO_EXCLUDE + [COMMA, START_TOKEN, END_TOKEN]

labels, img_paths = Dataset.load_paths_only(IMG_PATH)
labels_eval, img_paths_eval = Dataset.load_paths_only(IMG_PATH_EVAL)

generator = iter_sequence_infinite(DataGenerator(img_paths, labels, output_names, tokens_to_exclude, batch_size=32))
generator_eval = iter_sequence_infinite(
    DataGenerator(img_paths_eval, labels_eval, output_names, tokens_to_exclude, batch_size=32))

print("################################## MODEL ######################################")

new_model = CnnModelOriginal(output_names)

new_model.compile()

print("################################## TRAIN ######################################")

early_stopping = EarlyStopping(
    restore_best_weights=True,
    patience=10)

history = new_model.fit(generator,
                        validation_data=generator_eval,
                        epochs=10000,
                        steps_per_epoch=int(len(img_paths) / 32) * 8,
                        validation_steps=int(len(img_paths_eval) / 32),
                        callbacks=[early_stopping])
