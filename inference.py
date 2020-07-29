import io
import os
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms

def image_loader(loader, image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image

def get_pred(image_bytes, model):
    data_transforms = transforms.Compose([
         transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    pred = torch.tensor(model(image_loader(data_transforms, image_bytes)).detach().numpy())
    probs = F.softmax(pred).tolist()[0]
    return probs[0]