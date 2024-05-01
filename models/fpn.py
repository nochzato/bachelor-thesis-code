from models.crater_model import CraterModel


class FPN:
    def __init__(self, loss_fn, encoder_name="resnet34", lr=2e-4):
        self.model = CraterModel(
            arch="FPN",
            encoder_name=encoder_name,
            in_channels=3,
            out_classes=1,
            loss_fn=loss_fn,
            lr=lr,
        )

    def get_model(self):
        return self.model
