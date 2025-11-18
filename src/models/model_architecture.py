from src.data.csiro_dataset import CsiroDataset 
import pytorch_lightning as pl 
import timm
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from src.utils.r2_score import weighted_r2_score

class Csiro_model_Architecture(pl.LightningModule): 
    def __init__(
            self ,
            model_name = "efficientnet_b0" ,
            pretrained = False , 
            lr = 1e-4 , 
            output_dim = 5 ,
    ) : 
        super().__init__()  
        self.save_hyperparameters()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=output_dim) 
        self.criterion = nn.SmoothL1Loss() 
        self.val_outputs = [] 

    def forward(self , x): 
        return self.model(x)
    
    def training_step(self , batch , batch_idx):  
        x , y = batch 
        y_hat = self(x) 
        loss = self.criterion(y_hat , y)
        self.log("train_loss", loss , on_step=False , on_epoch =True) 
        return loss  
    
    def validation_step(self , batch , batch_idx): 
        x , y = batch 
        y_hat = self(x) 
        loss = self.criterion(y_hat , y) 
        self.val_outputs.append((y_hat.detach().cpu(), y.detach().cpu()))
        self.log("val_loss" , loss , on_step=False , on_epoch=True)  
        return loss 
    
    def on_validation_epoch_end(self):
        if len(self.val_outputs) == 0:
            self.log("val_weighted_r2", 0.0, prog_bar=True, on_epoch=True)
            for name in ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]:
                self.log(f"val_r2_{name}", 0.0, on_epoch=True)
            self.val_outputs.clear()
            return
        preds, trues = zip(*self.val_outputs)
        # preds/trues may be in bfloat16 when using bf16-mixed precision (macOS spawn).
        # Convert to float32 on CPU before converting to numpy to avoid unsupported dtype errors.
        preds = torch.cat(preds).to(torch.float32).numpy()
        trues = torch.cat(trues).to(torch.float32).numpy()
        weighted_r2, r2s = weighted_r2_score(trues, preds)
        self.log("val_weighted_r2", weighted_r2, prog_bar=True, on_epoch=True)
        for i, name in enumerate(["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]):
            self.log(f"val_r2_{name}", r2s[i], on_epoch=True)
        self.val_outputs.clear()
        return

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}