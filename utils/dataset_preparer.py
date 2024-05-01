import glob
import os
import shutil

import numpy as np
import tifffile


class DatasetPreparer:
    def __init__(self, IMG_DIM, NUM_SAMPLES=1000, MASK_PERCENTAGE=0.1):
        self.IMG_DIM = IMG_DIM
        self.NUM_SAMPLES = NUM_SAMPLES
        self.MASK_PERCENTAGE = MASK_PERCENTAGE

        self.img = tifffile.imread("./dataset/geo.tif")
        self.mask = tifffile.imread("./dataset/mask.tif")

        self.CLIP_IMG_DIR = f"./dataset/geo_{IMG_DIM}"
        self.CLIP_MASK_DIR = f"./dataset/mask_{IMG_DIM}"

        self.IMG_TRAIN_DIR = f"./dataset/geo_train_{IMG_DIM}"
        self.MASK_TRAIN_DIR = f"./dataset/mask_train_{IMG_DIM}"

        self.IMG_VAL_DIR = f"./dataset/geo_val_{IMG_DIM}"
        self.MASK_VAL_DIR = f"./dataset/mask_val_{IMG_DIM}"

        self.IMG_TEST_DIR = f"./dataset/geo_test_{IMG_DIM}"
        self.MASK_TEST_DIR = f"./dataset/mask_test_{IMG_DIM}"

    def prepare_dataset(self):
        self.__clip_data()
        self.__split_data()

    def __clip_data(self):
        self.__prepare_dirs([self.CLIP_IMG_DIR, self.CLIP_MASK_DIR])

        x_size = self.IMG_DIM
        y_size = self.IMG_DIM

        i = 0

        while i < self.NUM_SAMPLES:
            x = np.random.randint(0, self.img.shape[0] - x_size)
            y = np.random.randint(0, self.img.shape[1] - y_size)

            patch_img = self.img[x : x + x_size, y : y + y_size]
            patch_mask = self.mask[x : x + x_size, y : y + y_size]

            if (
                np.sum(patch_mask) / (self.IMG_DIM * self.IMG_DIM)
                > self.MASK_PERCENTAGE
            ):
                tifffile.imwrite(
                    os.path.join(self.CLIP_IMG_DIR, f"{i}_0.tif"), patch_img
                )
                tifffile.imwrite(
                    os.path.join(self.CLIP_MASK_DIR, f"{i}_mask.tif"), patch_mask
                )
                i += 1

    def __split_data(self):
        self.__prepare_dirs(
            [
                self.IMG_TRAIN_DIR,
                self.MASK_TRAIN_DIR,
                self.IMG_VAL_DIR,
                self.MASK_VAL_DIR,
                self.IMG_TEST_DIR,
                self.MASK_TEST_DIR,
            ]
        )

        images = glob.glob(os.path.join(self.CLIP_IMG_DIR, "*.tif"))
        masks = glob.glob(os.path.join(self.CLIP_MASK_DIR, "*.tif"))

        np.random.shuffle(images)

        train_size = int(0.8 * len(images))
        val_size = int(0.1 * len(images))

        train_images = images[:train_size]
        val_images = images[train_size : train_size + val_size]
        test_images = images[train_size + val_size :]

        for i in range(len(train_images)):
            shutil.copy(images[i], self.IMG_TRAIN_DIR)

            image_id = self.__get_image_id(images[i])
            mask = [m for m in masks if self.__get_image_id(m) == image_id][0]

            shutil.copy(mask, self.MASK_TRAIN_DIR)

        for i in range(len(val_images)):
            shutil.copy(images[i], self.IMG_VAL_DIR)

            image_id = self.__get_image_id(images[i])
            mask = [m for m in masks if self.__get_image_id(m) == image_id][0]

            shutil.copy(mask, self.MASK_VAL_DIR)

        for i in range(len(test_images)):
            shutil.copy(images[i], self.IMG_TEST_DIR)

            image_id = self.__get_image_id(images[i])
            mask = [m for m in masks if self.__get_image_id(m) == image_id][0]

            shutil.copy(mask, self.MASK_TEST_DIR)

    def __get_image_id(self, image_path):
        return os.path.normpath(image_path).split(os.sep)[-1].split("_")[0]

    def __prepare_dir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

        for f in glob.glob(dir + "/*"):
            os.remove(f)

    def __prepare_dirs(self, dirs):
        for dir in dirs:
            self.__prepare_dir(dir)
