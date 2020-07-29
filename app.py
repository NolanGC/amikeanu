import os
import cv2
import torch
import io
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.transforms as transforms
from flask import Flask, request, render_template
app = Flask(__name__) 

from inference import get_pred

app.config["UPLOAD_FOLDER"] = os.path.join("static", "images")
#from keanumodel import KeanuCnnModel

#pth_path = os.path.join(os.path.join("static", "models"), "keanu_resnet.pth")   
#model.load_state_dict(torch.load(pth_path)["state_dict"])
#model = torch.load(pth_path)
#torch.save(model, "model2.pth")
#model = KeanuCnnModel()
#model.load_state_dict(torch.load('cnnstatedict.pth'))
#model = torch.load('model2.pth')
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
        image = file.read()
        pred = get_pred(image, model)
        filename = "keanu.gif"
        return render_template(
            "result.html",
            #pred=str(round(pred*100,2)),
            pred=str(round(pred*100,2)),
            #otherpred = str(round(mathpred*100,2)),
            image=os.path.join(app.config["UPLOAD_FOLDER"], filename),
        )
if __name__ == "__main__":
    app.run(debug=True)
