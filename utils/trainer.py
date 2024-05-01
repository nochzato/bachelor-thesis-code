import pytorch_lightning as pl


class CraterTrainer:
    def __init__(self, logger):
        self.trainer = pl.Trainer(
            accelerator="gpu",
            max_epochs=1,
            logger=logger,
        )

    def get_trainer(self):
        return self.trainer
