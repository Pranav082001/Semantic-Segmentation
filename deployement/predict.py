import torch
import torchvision
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from model import DoubleConv,UNET

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


convert_tensor = transforms.ToTensor()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

model = UNET(in_channels=3, out_channels=1).to(device)
model=torch.load("Unet_acc_94.pth",map_location=torch.device('cpu'))

# test_img=np.array(Image.open("profilepic - Copy.jpeg").resize((160,240)))
test_img=Image.open("104.jpg").resize((240,160))

# test_img=torch.tensor(test_img).permute(2,1,0)
# test_img=test_img.unsqueeze(0)
test_img=convert_tensor(test_img).unsqueeze(0)
print(test_img.shape)
preds=model(test_img.float())
preds=torch.sigmoid(preds)
preds=(preds > 0.5).float()
print(preds.shape)
im=preds.squeeze(0).permute(1,2,0).detach()
print(im.shape)
fig,axs=plt.subplots(1,2)

axs[0].imshow(im)
axs[1].imshow(test_img.squeeze(0).permute(1,2,0).detach())
plt.show()
