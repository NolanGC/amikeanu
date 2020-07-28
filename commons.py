import io
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image
import torchvision.transforms as transforms
#from keanumodel import KeanuModel

def get_model():
    pass
    #return model

def get_tensor(image_bytes):
    my_transforms = transforms.ToTensor()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    open_cv_image = np.array(pil_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    new_array = cv2.resize(open_cv_image, (32, 32))
    #TODO TRY ORIGINAL MODEL BUT WITH THE LINE MARKED $
    new_array = np.array(new_array[:, :, [2, 1, 0]])
    new_array = np.transpose(new_array, (2, 0, 1)) #RGB AGAINherok
    img_tensor = torch.tensor(new_array, dtype=torch.float)
    print(img_tensor.shape)
    return img_tensor

