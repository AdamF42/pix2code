from cnn.CnnModel import CnnModel
from cnn.generator import DataGenerator
from utils.utils import load_pickle, eval_cnn_model
from utils.costants import TOKEN_TO_EXCLUDE, COMMA, START_TOKEN, END_TOKEN
from utils.dataset import Dataset

IMG_PATH = '../datasets/web/prove'

output_names = load_pickle('../pickle/output_names.pickle')

print("################################## DATA ###################################")

tokens_to_exclude = TOKEN_TO_EXCLUDE + [COMMA, START_TOKEN, END_TOKEN]
labels, img_paths = Dataset.load_paths_only(IMG_PATH)
generator = DataGenerator(img_paths, labels, output_names, tokens_to_exclude, batch_size=1)

data = [generator.__getitem__(i) + ({'name': img_paths[i]},) for i in range(len(img_paths))]

print("################################## MODEL ##################################")

new_model = CnnModel(output_names, image_out=True, dense_layer_size=512, dropout_ratio=0.1)
new_model.compile()
img_build, _ = generator.__getitem__(0)
new_model.predict(img_build)
new_model.load_weights('../instances/best_model_cnn_final.h5')

print("################################## EVAL ##################################")

train_errors, train_y, train_predictions = eval_cnn_model(new_model, data, output_names)
