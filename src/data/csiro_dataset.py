from torch.utils.data import Dataset     
from PIL import Image
import pandas as pd
import os
import torch

class CsiroDataset(Dataset): 
    def __init__(
            self,   
            df , 
            root_dir ,  
            transforms = None ,  
    ): 
        self.transforms = transforms  
        self.df = df  
        self.root_dir = root_dir
        self.train_df = pd.read_csv(os.path.join(root_dir, "train.csv")) 

    def __len__(self): 
        return len(self.df)
    
    def __getitem__(self , idx): 
        row = self.df.iloc[idx]  
        image = Image.open(os.path.join(self.root_dir, row["image_path"])).convert("RGB")  

        if self.transforms: 
            image = self.transforms(image) 

        targets = torch.tensor([
            row["Dry_Green_g"],
            row["Dry_Dead_g"],
            row["Dry_Clover_g"],
            row["GDM_g"],
            row["Dry_Total_g"]
        ], dtype=torch.float32) 

        return image, targets 