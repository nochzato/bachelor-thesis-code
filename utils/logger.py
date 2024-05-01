from lightning.pytorch.loggers import TensorBoardLogger


class CraterLogger:
    def __init__(self, logger_name):
        self.logger = TensorBoardLogger("logs", name=logger_name, log_graph=True)

    def get_logger(self):
        return self.logger
