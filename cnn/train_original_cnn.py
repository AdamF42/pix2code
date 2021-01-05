from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.utils.data_utils import iter_sequence_infinite

from CnnModelOriginal import CnnModelOriginal
from cnn.CnnModel import CnnModel
from generator import DataGenerator
from utils.utils import load_pickle, save_pickle, get_output_names
from utils.costants import TOKEN_TO_EXCLUDE, END_TOKEN, START_TOKEN, COMMA, PLACEHOLDER
from utils.dataset import Dataset

IMG_PATH = '../datasets/web/train_features'
IMG_PATH_EVAL = '../datasets/web/eval_features'

words = load_pickle('../pickle/output_names.pickle')
words = words + [COMMA, START_TOKEN, END_TOKEN, PLACEHOLDER]
print(len(words))
output_names = get_output_names(words)

print("################################# GENERATORS ##################################")

tokens_to_exclude = TOKEN_TO_EXCLUDE

labels, img_paths = Dataset.load_paths_only(IMG_PATH)
labels_eval, img_paths_eval = Dataset.load_paths_only(IMG_PATH_EVAL)

generator_samples = load_pickle('../pickle/cnn_train_generator_samples.pickle')
generator_eval_samples = load_pickle('../pickle/cnn_val_generator_samples.pickle')

generator = iter_sequence_infinite(DataGenerator(img_paths, labels, output_names, tokens_to_exclude,
                                                 samples=generator_samples, batch_size=32))
generator_val = iter_sequence_infinite(DataGenerator(img_paths_eval, labels_eval, output_names,
                                                      tokens_to_exclude, samples=generator_eval_samples,
                                                      batch_size=32))

# generator_samples = save_pickle(generator.samples,'../pickle/cnn_train_generator_samples.pickle')
# generator_eval_samples = save_pickle(generator_eval.samples, '../pickle/cnn_val_generator_samples.pickle')

print("################################## MODEL ######################################")

new_model = CnnModel(output_names, dense_layer_size=512, dropout_ratio=0.1)

new_model.compile()

print("################################## TRAIN ######################################")

checkpoint_filepath = '/content/gdrive/MyDrive/Colab Notebooks/CnnImageModel.h5'

import tensorflow as tf

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    save_best_only=True)

early_stopping = EarlyStopping(
    restore_best_weights=True,
    patience=10)

history = new_model.fit(generator,
                        validation_data=generator_val,
                        epochs=10000,
                        steps_per_epoch=int(len(img_paths) / 32) * 8,
                        validation_steps=int(len(img_paths_eval) / 32),
                        callbacks=[early_stopping])
