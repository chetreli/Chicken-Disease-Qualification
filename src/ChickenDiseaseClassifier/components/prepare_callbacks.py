from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import time

from ChickenDiseaseClassifier.entity.config_entity import PrepareCallbacksConfig

class PrepareCallback:
    def __init__(self, config: PrepareCallbacksConfig):
        self.config = config

    def get_tb_ckpt_callbacks(self):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")

        tb_logger = TensorBoardLogger(
            save_dir=self.config.tensorboard_root_log_dir,
            name=f"tb_logs_at_{timestamp}"
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config.root_dir,
            filename="best-checkpoint",
            save_top_k=1,
            monitor="val_loss",
            mode="min"
        )

        return tb_logger, checkpoint_callback