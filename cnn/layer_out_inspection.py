from cnn.CnnModel import CnnModel
from cnn.generator import DataGenerator
from utils.utils import load_pickle, inspect_layer
from w2v_test.costants import TOKEN_TO_EXCLUDE, COMMA, START_TOKEN, END_TOKEN
from utils.dataset import Dataset

output_names = load_pickle('../pickle/output_names.pickle')

IMG_PATH = '../datasets/web/prove'

tokens_to_exclude = TOKEN_TO_EXCLUDE + [COMMA, START_TOKEN, END_TOKEN]
labels, img_paths = Dataset.load_paths_only(IMG_PATH)
generator = DataGenerator(img_paths, labels, output_names, tokens_to_exclude, batch_size=1)

new_model = CnnModel(output_names, image_out=True, dense_layer_size=512, dropout_ratio=0.1)
new_model.compile()
img_build, _ = generator.__getitem__(0)
new_model.predict(img_build)

new_model.load_weights('../instances/best_model_cnn_final.h5')

data = [generator.__getitem__(i) + ({'name': img_paths[i]},) for i in range(len(img_paths))]

for img, _, name in data:
    inspect_layer(new_model, img, name, output_names)
