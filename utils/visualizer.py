import matplotlib.pyplot as plt
import torch


class CraterVisualizer:
    def __init__(self, model, test_dataloader):
        self.model = model
        self.test_dataloader = test_dataloader

    def visualize(self):
        batch = next(iter(self.test_dataloader))
        with torch.no_grad():
            self.model.eval()
            logits = self.model(batch["image"])
        pr_masks = logits.sigmoid()

        for image, gt_mask, pr_mask in zip(batch["image"], batch["mask"], pr_masks):
            plt.figure(figsize=(10, 5))

            plt.subplot(1, 3, 1)
            plt.imshow(image.numpy().transpose(1, 2, 0))
            plt.title("Image")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(gt_mask.numpy().squeeze())
            plt.title("Ground truth")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(pr_mask.numpy().squeeze())
            plt.title("Prediction")
            plt.axis("off")

            plt.show()
