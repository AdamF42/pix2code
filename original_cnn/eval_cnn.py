from functools import reduce

from tensorflow.python.keras.models import load_model

from original_cnn.Pix2CodeOriginalCnnModel import Pix2CodeOriginalCnnModel
from original_cnn.generator import DataGenerator
from w2v_test.costants import TOKEN_TO_EXCLUDE, COMMA, START_TOKEN, END_TOKEN, CNN_OUTPUT_NAMES
from w2v_test.dataset.dataset import Dataset

# IMG_W2V_TRAIN_DIR = '/home/adamf42/Projects/pix2code/datasets/web/eval_set'
from w2v_test.dataset.utils import get_token_sequences_with_max_seq_len

IMG_PATH = '/home/adamf42/Projects/pix2code/datasets/web/prove'

print("################################## MODEL ##################################")


tokens_to_exclude = TOKEN_TO_EXCLUDE + [COMMA, START_TOKEN, END_TOKEN]

tokens = get_token_sequences_with_max_seq_len(IMG_PATH, tokens_to_exclude)

voc = {token for token in reduce(lambda x, y: x + y, tokens['sentences'])}
output_names = list(map(lambda x: CNN_OUTPUT_NAMES[x] if x in CNN_OUTPUT_NAMES.keys() else x, voc))

new_model = Pix2CodeOriginalCnnModel(output_names)

new_model.compile()

# new_model: Pix2CodeOriginalCnnModel = load_model('pix2code_cnn_china')
new_model: Pix2CodeOriginalCnnModel = load_model('pix2code_cnn')

print("################################## EVAL ##################################")

# infinite loop
labels, img_paths = Dataset.load_paths_only(IMG_PATH)

generator = DataGenerator(img_paths, labels, output_names, tokens_to_exclude, batch_size=1)

data = [generator.__getitem__(i) for i in range(len(img_paths))]

errors = {}
for img, label in data:
    prediction = new_model.predict(img)
    for key in label.keys():
        if not prediction.get(key) or label[key][0] != int(round(prediction[key][0][0])):
            if errors.get(key):
                errors[key] = errors[key] + 1
            else:
                errors[key] = 1
print(errors)
