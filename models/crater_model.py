import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch


class CraterModel(pl.LightningModule):
    def __init__(
        self, arch, encoder_name, in_channels, out_classes, loss_fn, lr, **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )

        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.loss_fn = loss_fn
        self.learning_rate = lr

    def forward(self, image):
        # Normalize the image
        image = (image - self.mean) / self.std
        mask = self.model(image.float())

        return mask

    def shared_step(self, batch, step):
        image = batch["image"]

        assert image.ndim == 4

        h, w = image.shape[2:]

        assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask"]

        assert mask.ndim == 4

        logits_mask = self.forward(image)

        loss = self.loss_fn(logits_mask, mask)

        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode="binary"
        )

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        average_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )

        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        if stage == "train":
            self.training_step_outputs = []
        elif stage == "val":
            self.validation_step_outputs = []
        elif stage == "test":
            self.test_step_outputs = []

        metrics = {
            f"{stage}_loss": average_loss,
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        out = self.shared_step(batch, "train")
        self.training_step_outputs.append(out)
        return out

    def on_train_epoch_end(self):
        return self.shared_epoch_end(outputs=self.training_step_outputs, stage="train")

    def validation_step(self, batch, batch_idx):
        out = self.shared_step(batch, "val")
        self.validation_step_outputs.append(out)
        return out

    def on_validation_epoch_end(self):
        return self.shared_epoch_end(outputs=self.validation_step_outputs, stage="val")

    def on_test_epoch_start(self):
        self.logger.log_hyperparams(self.hparams)

    def test_step(self, batch, batch_idx):
        out = self.shared_step(batch, "test")
        self.test_step_outputs.append(out)
        return out

    def on_test_epoch_end(self):
        return self.shared_epoch_end(outputs=self.test_step_outputs, stage="test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
