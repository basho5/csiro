import sys
import os
# Ensure project root (one level above `src`) is on sys.path so `import src...` works
# train.py is at: <project>/src/models/train.py
# os.path.dirname(__file__) -> <project>/src/models
# join(..., '..', '..') -> <project>
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import KFold
from pytorch_lightning.callbacks import ModelCheckpoint
from src.data.csiro_datamodule import Csiro_Data_Module
from src.models.model_architecture import (
    Csiro_model_Architecture 
)
from src.config.config_dataset import config_dataset


def main():
    # train_df = pd.read_csv('/kaggle/input/csiro-biomass/train.csv')
    train_df = pd.read_csv(os.path.join(config_dataset.root_dir, 'train.csv'))
    train_df = pd.pivot_table(train_df, index='image_path', columns=['target_name'], values='target').reset_index()

    kf = KFold(n_splits=5, random_state=0, shuffle=True)

    for fold, (train_index, valid_index) in enumerate(kf.split(train_df)):
        print(f"\n{'='*50}")
        print(f"Training Fold {fold + 1}/5")
        print(f"{'='*50}\n")
        
        datamodule = Csiro_Data_Module(train_df.iloc[train_index], train_df.iloc[valid_index])
        model = Csiro_model_Architecture(model_name="efficientnet_b2", pretrained=True, lr=1e-4)
        
        checkpoint_callback = ModelCheckpoint(
            monitor="val_weighted_r2",
            save_top_k=1,
            mode="max",
            filename=f"best_model_fold{fold}"
        )
        
        # Disable TensorBoard logger to avoid import errors
        trainer = pl.Trainer(
            max_epochs=10,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            callbacks=[checkpoint_callback],
            precision="16-mixed",
            logger=False
        )
        
        trainer.fit(model, datamodule=datamodule)
        torch.save(model.state_dict(), f"model_fold{fold}.pth")
        print(f"\nFold {fold + 1} training complete!")


if __name__ == '__main__':
    # On platforms using 'spawn' (macOS, default), protect the entrypoint so child
    # processes don't re-run the top-level training code when they import this module.
    try:
        # multiprocessing.freeze_support is a no-op on Unix but required on Windows; safe to call.
        from multiprocessing import freeze_support
        freeze_support()
    except Exception:
        pass
    main()
