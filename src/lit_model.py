import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from src.utils import get_logger

log = get_logger(__name__)


class BaseLitModel(pl.LightningModule):
    def __init__(
        self,
        models_config: DictConfig,
        optimizer_config: DictConfig,
        scheduler_config: DictConfig,
        use_weights_path: str,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        # Instantiate a model with random weights, or load them from a checkpoint
        if self.hparams.use_weights_path is None:
            for model_name, model_config in self.hparams.models_config.items():
                model = hydra.utils.instantiate(model_config)
                setattr(self, model_name, model)
        else:
            ckpt = BaseLitModel.load_from_checkpoint(self.hparams.use_weights_path)

            for model_name in ckpt.hparams.models_config:
                model = getattr(ckpt, model_name)
                setattr(self, model_name, model)

        self.optimizer = hydra.utils.instantiate(self.hparams.optimizer_config, params=self.parameters())
        self.scheduler = (
            hydra.utils.instantiate(self.hparams.scheduler_config, optimizer=self.optimizer)
            if self.hparams.scheduler_config is not None
            else None
        )

    def configure_optimizers(self):
        if self.scheduler is None:
            return self.optimizer

        return [self.optimizer], [{"scheduler": self.scheduler, "interval": "epoch"}]


class ExampleMNISTLitModel(BaseLitModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def training_step(self, batch, _) -> torch.Tensor:
        x, y = batch

        pred = self.model(x)
        loss = F.cross_entropy(pred, y)

        accuracy = (pred.max(dim=1).indices == y).float().mean()

        self.log("train_accuracy", accuracy, prog_bar=True, logger=True)
        self.log("train_loss", loss, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, _) -> None:
        x, y = batch

        pred = self.model(x)
        loss = F.cross_entropy(pred, y)

        accuracy = (pred.max(dim=1).indices == y).float().mean()

        self.log("val_accuracy", accuracy, prog_bar=True, logger=True)
        self.log("val_loss", loss, prog_bar=True, logger=True)
