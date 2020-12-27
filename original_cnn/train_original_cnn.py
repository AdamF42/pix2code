from functools import reduce

from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.utils.data_utils import iter_sequence_infinite

from Pix2CodeOriginalCnnModel import Pix2CodeOriginalCnnModel
from generator import DataGenerator
from w2v_test.dataset.dataset import Dataset
from w2v_test.costants import TOKEN_TO_EXCLUDE, END_TOKEN, START_TOKEN, COMMA, CNN_OUTPUT_NAMES
from w2v_test.dataset.utils import get_token_sequences_with_max_seq_len

IMG_W2V_TRAIN_DIR = '/home/adamf42/Projects/pix2code/datasets/web/all_data'
IMG_PATH = '/home/adamf42/Projects/pix2code/datasets/web/training_features'
IMG_PATH_EVAL = '/home/adamf42/Projects/pix2code/datasets/web/training_features'

tokens_to_exclude = TOKEN_TO_EXCLUDE + [COMMA, START_TOKEN, END_TOKEN]

tokens = get_token_sequences_with_max_seq_len(IMG_W2V_TRAIN_DIR, tokens_to_exclude)
voc = {token for token in reduce(lambda x, y: x + y, tokens['sentences'])}
output_names = list(map(lambda x: CNN_OUTPUT_NAMES[x] if x in CNN_OUTPUT_NAMES.keys() else x, voc))

new_model = Pix2CodeOriginalCnnModel(output_names)

new_model.compile()

# shape = (None, 256, 256, 3)

# new_model.build(shape)

# new_model.summary()

# infinite loop
labels, img_paths = Dataset.load_paths_only(IMG_PATH)
labels_eval, img_paths_eval = Dataset.load_paths_only(IMG_PATH_EVAL)

generator = iter_sequence_infinite(DataGenerator(img_paths, labels, output_names, tokens_to_exclude, batch_size=32))
generator_eval = iter_sequence_infinite(DataGenerator(img_paths_eval, labels_eval, output_names, tokens_to_exclude, batch_size=32))


early_stopping = EarlyStopping(
    restore_best_weights=True,
    patience=10)

history = new_model.fit(generator,
                            validation_data=generator_eval,
                            epochs=10000,
                            steps_per_epoch=int(len(img_paths)/32)*8,
                            validation_steps=int(len(img_paths_eval)/32)*8,
                            callbacks=[early_stopping])
