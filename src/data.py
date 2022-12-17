import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms


def get_mnist_dataset(data_dir: str):
    transform = transforms.ToTensor()
    train_set = MNIST(data_dir, train=True, transform=transform, download=True)
    val_set = MNIST(data_dir, train=False, transform=transform, download=True)

    return train_set, val_set


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset_config: DictConfig, batch_size: int, shuffle: bool = True, num_workers: int = 1):
        super().__init__()

        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_set, self.val_set = hydra.utils.instantiate(self.dataset_config)

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
