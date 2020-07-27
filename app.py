import os
import cv2
import torch
import io
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from keanumodel import KeanuModel
import torchvision.transforms as transforms
from flask import Flask, request, render_template
app = Flask(__name__) 

from commons import get_tensor
from inference import get_pred

app.config["UPLOAD_FOLDER"] = os.path.join("static", "images")

class KeanuModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3 * 64 * 64, 32)
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, 2)

    def forward(self, xb):
        xb = xb.view(xb.size(0), -1)
        out = self.linear1(xb)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = self.linear4(out)
        return out

@app.route("/", methods=["GET", "POST"])
def hello_world():
    if request.method == "GET":
        return render_template("index.html", value="hi")
    if request.method == "POST":
        if "file" not in request.files:
            print("no file brub")
            return
        file = request.files["file"]
        image = file.read()
        pred = get_pred(image)
        filename = "keanupoint.jpg"
        return render_template(
            "result.html",
            pred=str(round(pred * 100, 2)),
            image=os.path.join(app.config["UPLOAD_FOLDER"], filename),
        )
if __name__ == "__main__":
    model = KeanuModel()
    KeanuModel.__module__ = "keanumodel"
    model.load_state_dict(torch.load(os.path.join(os.path.join("static", "images"), "checkpoint.pth"))["state_dict"])
    app.run(debug=True)
