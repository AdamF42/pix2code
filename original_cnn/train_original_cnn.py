from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.utils.data_utils import iter_sequence_infinite


from original_cnn.Pix2CodeOriginalCnnModel import Pix2CodeOriginalCnnModel
from original_cnn.costants import TOKENS_TO_INDEX
from original_cnn.generator import DataGenerator
from w2v_test.dataset.dataset import Dataset

IMG_PATH = '/home/adamf42/Projects/pix2code/datasets/web/single'
IMG_PATH_EVAL = '/home/adamf42/Projects/pix2code/datasets/web/single'

voc = list(TOKENS_TO_INDEX.keys())
output_names=[]
names = map(lambda x: "open_bracket" if x == "{" else x, voc)
names = map(lambda x: "close_bracket" if x == "}" else x, names)
names = map(lambda x: "comma" if x == "," else x, names)
for name in names:
    output_names.append(name)

# print(output_names)
# print(len(output_names))

new_model = Pix2CodeOriginalCnnModel(output_names)

new_model.compile()


# shape = (None, 256, 256, 3)

# new_model.build(shape)

# new_model.summary()

# infinite loop
labels, img_paths = Dataset.load_paths_only(IMG_PATH)
labels_eval, img_paths_eval = Dataset.load_paths_only(IMG_PATH_EVAL)

generator = iter_sequence_infinite(DataGenerator(img_paths, labels, output_names, batch_size=32))
generator_eval = iter_sequence_infinite(DataGenerator(img_paths_eval, labels_eval, output_names, batch_size=32))

# X,y = generator.__getitem__(0)
# print(X,y)
# new_model.fit(generator, epochs=10)

# import matplotlib.pyplot as plt
# import numpy


early_stopping = EarlyStopping(
    restore_best_weights=True,
    patience=10)

history = new_model.fit(generator,
                            validation_data=generator_eval,
                            epochs=10000,
                            steps_per_epoch=int(len(img_paths)/32)*8,
                            validation_steps=int(len(img_paths_eval)/32)*8,
                            callbacks=[early_stopping])
