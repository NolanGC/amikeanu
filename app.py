import io
import os
import cv2
import math
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from flask import Flask, request, render_template

app = Flask(__name__) 

from inference import get_pred

app.config["UPLOAD_FOLDER"] = os.path.join("static", "images")

model = torch.load('model.pth')
model.eval()

@app.route("/", methods=["GET", "POST"])
def site():
    if request.method == "GET":
        return render_template("index.html")
    if request.method == "POST":
        if "file" not in request.files:
            print("no files")
            return
        file = request.files["file"]
        pred = get_pred(file.read(), model)
        return render_template(
            "result.html",
            pred=str(round(pred*100,2)),
            image=os.path.join(app.config["UPLOAD_FOLDER"], "keanu.gif"),
        )
if __name__ == "__main__":
    app.run(debug=True)