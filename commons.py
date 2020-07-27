import io
import os
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from keanumodel import KeanuModel

def get_model():
    pth_path = os.path.join(os.path.join("static", "images"), "checkpoint.pth")
    model = KeanuModel()
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

