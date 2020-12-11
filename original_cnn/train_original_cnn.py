from original_cnn.Pix2CodeOriginalCnnModel import Pix2CodeOriginalCnnModel
from original_cnn.costants import TOKENS_TO_INDEX
from original_cnn.generator import DataGenerator
from w2v_test.dataset.dataset import Dataset
import tensorflow as tf

IMG_PATH = '/home/adamf42/Projects/pix2code/datasets/web/prove'

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


shape = (None, 256, 256, 3)

# new_model.build(tf.TensorShape([256, 256, 3]))
#
# new_model.summary()

labels, img_paths = Dataset.load_paths_only(IMG_PATH)

generator = DataGenerator(img_paths, labels, output_names, batch_size=2)

X,y = generator.__getitem__(0)

# print(type(X))
# print(X, y)

# output_types_x = {'img_data': tf.float32}
# output_shapes_x = {'img_data': tf.TensorShape([256, 256, 3])}
# output_types_y = {}
# output_shapes_y = {}
#
# output_types_y.update({key + "_count": tf.float32 for key in output_names})
# output_shapes_y.update({key + "_count": tf.TensorShape([]) for key in output_names})

# test = tf.data.Dataset.from_generator(generator,
#                                       output_types=(output_types_x, output_types_y),
#                                       output_shapes=(output_shapes_x, output_shapes_y)
#                                       )
new_model.fit(generator, epochs=10)
