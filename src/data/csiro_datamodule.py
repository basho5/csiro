import pytorch_lightning as pl  
import torchvision.transforms as T  
from src.data.csiro_dataset import CsiroDataset  
from torch.utils.data import DataLoader 
from src.config.config_dataset import config_dataset

class Csiro_Data_Module(pl.LightningDataModule):
    def __init__(self, train_df, valid_df, batch_size=8, num_workers=4, img_size=1000):
        super().__init__()
        self.train_df = train_df
        self.valid_df = valid_df
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size

    def setup(self, stage=None):
        train_transform = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        val_transform = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.train_dataset = CsiroDataset(self.train_df,config_dataset.root_dir, train_transform)
        self.val_dataset = CsiroDataset(self.valid_df,config_dataset.root_dir, val_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)



