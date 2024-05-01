import os

from torch.utils.data import DataLoader

from utils.dataset import CraterDataset


class CraterDataloaders:
    def __init__(self, IMG_DIM):
        train_dataset = CraterDataset(
            img_dir=f"./dataset/geo_train_{IMG_DIM}",
            mask_dir=f"./dataset/mask_train_{IMG_DIM}",
        )

        val_dataset = CraterDataset(
            img_dir=f"./dataset/geo_val_{IMG_DIM}",
            mask_dir=f"./dataset/mask_val_{IMG_DIM}",
        )
        test_dataset = CraterDataset(
            img_dir=f"./dataset/geo_test_{IMG_DIM}",
            mask_dir=f"./dataset/mask_test_{IMG_DIM}",
        )

        n_cpus = os.cpu_count()

        self.train_dataloader = DataLoader(
            train_dataset, batch_size=16, shuffle=True, num_workers=n_cpus
        )
        self.val_dataloader = DataLoader(
            val_dataset, batch_size=16, shuffle=False, num_workers=n_cpus
        )
        self.test_dataloader = DataLoader(
            test_dataset, batch_size=16, shuffle=False, num_workers=n_cpus
        )
