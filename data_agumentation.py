import glob
import os
import shutil

from keras_preprocessing.image import ImageDataGenerator, np

from utils.utils import get_preprocessed_img

image_transformer = ImageDataGenerator(horizontal_flip=True, width_shift_range=5, fill_mode='nearest', vertical_flip=True)

input_path = 'datasets/web/agumented_train_features'

img_list = list(glob.glob(os.path.join(input_path, "*.npz")))

for img in img_list:
    file_name = img[:img.find(".npz")].split('/')[-1]
    img_data = get_preprocessed_img(img)
    # img_data = img['img_data']
    img_data = image_transformer.random_transform(img_data)

    np.savez_compressed("{}/{}".format(input_path, 'agumented'+file_name), features=img_data)
    shutil.copyfile("{}/{}.gui".format(input_path, file_name), "{}/{}.gui".format(input_path, 'agumented'+file_name))