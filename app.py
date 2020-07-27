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
