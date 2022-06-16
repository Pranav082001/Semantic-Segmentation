import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import torch.nn as nn
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import transforms

from model import DoubleConv,UNET
    
convert_tensor = transforms.ToTensor()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNET(in_channels=3, out_channels=1).to(device)
# model=torch.load("Unet_acc_94.pth",map_location=torch.device('cpu'))

model=torch.load("Unet_acc_94.pth",map_location=device)

def predict(img):
    img=cv2.resize(img,(240,160))
    test_img=convert_tensor(img).unsqueeze(0)
    print(test_img.shape)
    preds=model(test_img.float())
    preds=torch.sigmoid(preds)
    preds=(preds > 0.5).float()
    print(preds.shape)
    im=preds.squeeze(0).permute(1,2,0).detach()
    print(im.shape)
    im=im.numpy()
    return im

import streamlit as st
st.title("Image Colorizer")
        
file=st.file_uploader("Please upload the B/W image",type=["jpg","jpeg","png"])
print(file)
if file is None:
  st.text("Please Upload an image")
else:
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    im=predict(opencv_image)
    st.text("Original")
    st.image(file)
    st.text("Colorized!!")
    st.image(im)



