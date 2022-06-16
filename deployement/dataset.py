import torch
from torch.utils.data.dataloader import DataLoader,Dataset
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

class Segmentation_Dataset(Dataset):
    def __init__(self,img_dir,mask_dir,transform=None):
        self.img_dir=img_dir
        self.mask_dir=mask_dir
        self.transform=transform
        self.images=os.listdir(img_dir)
        self.images=[im for im in self.images if ".jpg" in im]
    def __len__(self):
        return len(self.images)

    def __getitem__(self,idx):
        img_path=os.path.join(self.img_dir,self.images[idx])
        mask_path=os.path.join(self.mask_dir,self.images[idx].replace(".jpg",".png"))

        image=np.array(Image.open(img_path).convert("RGB"))
        mask=np.array(Image.open(mask_path).convert("L"),dtype=np.float32)
        mask[mask==255]=1.0

        if self.transform is not None:
            augmentations=self.transform(image=image,mask=mask)
            image=augmentations["image"]
            mask=augmentations["mask"]

        return image, mask
        