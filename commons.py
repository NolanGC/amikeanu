import io
import os
import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms
from keanumodel import KeanuModel
import cv2


def get_model():
    pth_path = "/Users/nolanclement/amikeanu/checkpoint.pth"
    model = KeanuModel(3 * 64 * 64, hidden_size=32, out_size=2)
    model.load_state_dict(torch.load(pth_path)["state_dict"])
    return model


def get_tensor(image_bytes):
    my_transforms = transforms.ToTensor()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    open_cv_image = np.array(pil_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    new_array = cv2.resize(open_cv_image, (64, 64))
    img_tensor = torch.tensor(new_array, dtype=torch.float)
    return img_tensor
