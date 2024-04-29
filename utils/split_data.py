import glob
import os
import shutil

import numpy as np

from utils.os_helper import prepare_dir

IMG_DIM = 128

IMG_DIR = f"./dataset/geo_{IMG_DIM}"
MASK_DIR = f"./dataset/mask_{IMG_DIM}"

IMG_TRAIN_DIR = f"./dataset/geo_train_{IMG_DIM}"
MASK_TRAIN_DIR = f"./dataset/mask_train_{IMG_DIM}"

IMG_VAL_DIR = f"./dataset/geo_val_{IMG_DIM}"
MASK_VAL_DIR = f"./dataset/mask_val_{IMG_DIM}"

IMG_TEST_DIR = f"./dataset/geo_test_{IMG_DIM}"
MASK_TEST_DIR = f"./dataset/mask_test_{IMG_DIM}"


prepare_dir(IMG_TRAIN_DIR)
prepare_dir(MASK_TRAIN_DIR)
prepare_dir(IMG_VAL_DIR)
prepare_dir(MASK_VAL_DIR)
prepare_dir(IMG_TEST_DIR)
prepare_dir(MASK_TEST_DIR)

images = glob.glob(os.path.join(IMG_DIR, "*.tif"))
masks = glob.glob(os.path.join(MASK_DIR, "*.tif"))

np.random.shuffle(images)

# Split the data into training, validation and testing
# with propotions 80%, 10%, 10%
train_size = int(0.8 * len(images))
val_size = int(0.1 * len(images))
test_size = len(images) - train_size - val_size

train_images = images[:train_size]
val_images = images[train_size : train_size + val_size]
test_images = images[train_size + val_size :]


def get_image_id(image_path):
    return image_path.split("/")[-1].split("\\")[1].split("_")[0]


def copy_images_and_masks(image, mask, img_dir, mask_dir):
    shutil.copy(image, img_dir)
    image_id = get_image_id(image)
    # Find the corresponding mask
    mask = [m for m in masks if get_image_id(m) == image_id][0]
    shutil.copy(mask, mask_dir)


for i in range(len(train_images)):
    copy_images_and_masks(train_images[i], masks[i], IMG_TRAIN_DIR, MASK_TRAIN_DIR)

for i in range(len(val_images)):
    copy_images_and_masks(val_images[i], masks[i], IMG_VAL_DIR, MASK_VAL_DIR)

for i in range(len(test_images)):
    copy_images_and_masks(test_images[i], masks[i], IMG_TEST_DIR, MASK_TEST_DIR)
