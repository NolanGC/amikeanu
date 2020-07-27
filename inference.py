from commons import get_model, get_tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from keanumodel import KeanuModel

model = get_model()

def get_pred(image_bytes):
    tensor = get_tensor(image_bytes)
    test_loader = DataLoader([(tensor, 0)], 1)
    for img, _ in test_loader:
        pred = model(img)
        break
    probs = F.softmax(pred, dim=1)[0].tolist()
    return probs[0]
    