from original_cnn.Pix2CodeOriginalCnnModel import Pix2CodeOriginalCnnModel
from original_cnn.costants import TOKENS_TO_INDEX
from original_cnn.generator import DataGenerator
from w2v_test.dataset.dataset import Dataset

IMG_PATH = '/home/adamf42/Projects/pix2code/datasets/web/training_features'

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

labels, img_paths = Dataset.load_paths_only(IMG_PATH)

generator = DataGenerator(img_paths, labels, output_names, batch_size=2)

# X,y = generator.__getitem__(0)

new_model.fit(generator, epochs=10)
