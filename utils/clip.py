import os

import numpy as np
import tifffile

from utils.os_helper import prepare_dir

IMG_DIM = 128

NUM_SAMPLES = 1000

IMG_DIR = f"./dataset/geo_{IMG_DIM}"
MASK_DIR = f"./dataset/mask_{IMG_DIM}"

prepare_dir(IMG_DIR)
prepare_dir(MASK_DIR)

img = tifffile.imread("./dataset/geo.tif")
mask = tifffile.imread("./dataset/mask.tif")


x_size = IMG_DIM
y_size = IMG_DIM


i = 0


while i < NUM_SAMPLES:
    x = np.random.randint(0, img.shape[0] - x_size)
    y = np.random.randint(0, img.shape[1] - y_size)

    patch_img = img[x : x + x_size, y : y + y_size]
    patch_mask = mask[x : x + x_size, y : y + y_size]

    # Save the patch if it contains at least 10% of the mask
    if np.sum(patch_mask) / (IMG_DIM * IMG_DIM) > 0.1:
        tifffile.imwrite(os.path.join(IMG_DIR, f"{i}_0.tif"), patch_img)
        tifffile.imwrite(os.path.join(MASK_DIR, f"{i}_mask.tif"), patch_mask)
        i += 1
