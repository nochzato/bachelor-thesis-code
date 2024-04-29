import os
import numpy as np

import tifffile
from torch.utils.data import Dataset


# Define the dataset class
# Images and correspoindng masks are images
class CraterDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        real_idx = os.listdir(self.img_dir)[idx].split("_")[0]

        img_path = os.path.join(self.img_dir, f"{real_idx}_0.tif")

        mask_path = os.path.join(self.mask_dir, f"{real_idx}_mask.tif")

        image = tifffile.imread(img_path)
        mask = tifffile.imread(mask_path)

        # Remove INR image channel
        image = image[:, :, :3]

        # Apply min-max scaling to the image
        image = (image - np.min(image)) / (np.max(image) - np.min(image))

        # Convert image from (H, W, C) to (C, H, W)
        image = np.transpose(image, (2, 0, 1))

        # Convert mask from (H, W) to (1, H, W)
        mask = np.expand_dims(mask, axis=0)

        return {"image": image, "mask": mask}
