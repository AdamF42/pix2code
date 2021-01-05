from cnn.CnnImageModel import CnnImageModel
from cnn.CnnModel import CnnModel
from cnn.CnnModelOriginal import CnnModelOriginal
from cnn.generator import DataGenerator
from utils.utils import load_pickle, eval_cnn_model, get_output_names
from utils.costants import TOKEN_TO_EXCLUDE, COMMA, START_TOKEN, END_TOKEN
from utils.dataset import Dataset

IMG_PATH = '../datasets/web/prove'

words = load_pickle('../pickle/output_names.pickle')
words = words + [COMMA, START_TOKEN, END_TOKEN]
output_names = get_output_names(words)
print(len(output_names))
tokens_sequences = load_pickle('../pickle/tokens_sequences_no_spaces.pickle')
max_sentence_len = tokens_sequences['max_sentence_len']

print("################################## DATA ###################################")

tokens_to_exclude = TOKEN_TO_EXCLUDE
labels, img_paths = Dataset.load_paths_only(IMG_PATH)
generator_eval_samples = load_pickle('../pickle/val_features.pickle')

generator = DataGenerator(img_paths, labels, output_names, tokens_to_exclude,
                          samples=generator_eval_samples, batch_size=1)

data = [generator.__getitem__(i) + ({'name': img_paths[i]},) for i in range(len(img_paths))]

print("################################## MODEL ##################################")

# new_model = CnnModel(output_names, image_out=True, dense_layer_size=512, dropout_ratio=0.1)
new_model = CnnImageModel(words, output_names,  max_sentence_length=max_sentence_len,
                          dense_layer_size=512, dropout_ratio=0.1)
new_model.compile()
img_build, label = generator.__getitem__(0)
# print(list(label.keys()))

# new_model.predict(img_build)
# new_model.load_weights('../instances/best_model_cnn_final.h5')
# new_model.load_weights('../instances/CnnImageModel.h5')

print("################################## EVAL ##################################")

train_errors, train_y, train_predictions = eval_cnn_model(new_model, data, output_names)

print(train_errors)