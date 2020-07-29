import io
import os
import cv2
import torch
from PIL import Image
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

# def get_tensor(image_bytes):
#     my_transforms = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
#     pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#     open_cv_image = np.array(pil_image)
#     open_cv_image = open_cv_image[:, :, ::-1].copy()
#     new_array = cv2.resize(open_cv_image, (224,224))
#     new_array = np.array(new_array[:, :, [2, 1, 0]])
#     new_array = np.transpose(new_array, (2, 0, 1))
#     img_tensor = torch.tensor(new_array, dtype=torch.float)
#     return img_tensor

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
    #print(model(F.softmax(image_loader(data_transforms, image_bytes),dim=1)[0]))
    pred = torch.tensor(model(image_loader(data_transforms, image_bytes)).detach().numpy())
    probs = F.softmax(pred).tolist()[0]
    return probs[0]
    # tensor = get_tensor(image_bytes)
    # print(tensor.shape)
    # test_loader = DataLoader([(tensor, 0)], 1)
    # for img, label in test_loader:
    #     outputs = model(img)
    #     _, preds = torch.max(outputs, 1)
    #     print("Label: ",label)
    # print("Softmax: ", F.softmax(outputs, dim=1))
    # print("preds; ", preds)
    #probs = F.softmax(pred, dim=1)[0].tolist()
    #print(probs)
    #print(probs)
    #return probs[1]
    