#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function

import glob
import os
import shutil
import sys
from math import floor

argv = sys.argv[1:]

if len(argv) < 2:
    print("Error: not enough argument supplied:")
    print("convert_imgs_to_arrays.py <input path> <output path>")
    exit(0)
else:
    input_path = argv[0]
    output_path = argv[1]


print(output_path)
os.makedirs(output_path + "train_features", exist_ok=True)
os.makedirs(output_path + "validation_features", exist_ok=True)

# print("Converting images to numpy arrays...")

img_list = list(glob.glob(os.path.join(input_path, "*.gui")))
img_list = list(map(lambda x: x.replace(".gui", ""), img_list))

# print(img_list)

for i in range(0, len(img_list)):
    if i < floor(len(img_list) * 0.9):
        shutil.copyfile("{}.gui".format(img_list[i]),
                        "{}/{}.gui".format(output_path + "train_features", img_list[i].split('/')[-1]))
        shutil.copyfile("{}.npz".format(img_list[i]),
                        "{}/{}.npz".format(output_path + "train_features", img_list[i].split('/')[-1]))
    else:
        shutil.copyfile("{}.gui".format(img_list[i]),
                        "{}/{}.gui".format(output_path + "validation_features", img_list[i].split('/')[-1]))
        shutil.copyfile("{}.npz".format(img_list[i]),
                        "{}/{}.npz".format(output_path + "validation_features", img_list[i].split('/')[-1]))
