from flask import Flask, request, render_template
from commons import get_tensor
from inference import getPred
import cv2
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from keanumodel import KeanuModel
from torch.utils.data import DataLoader

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
    
input_size = 3 * 64 * 64
num_classes = 2
hidden_size = 64
hidden_size2 = 32
hidden_size3 = 16

app = Flask(__name__) 
app.config["UPLOAD_FOLDER"] = os.path.join("static", "images")

@app.route("/", methods=["GET", "POST"])
def hello_world():
    if request.method == "GET":
        return render_template("index.html", value="hi")
    if request.method == "POST":
        print(request.files)
        if "file" not in request.files:
            print("no file brub")
            return
        file = request.files["file"]
        image = file.read()
        tensor = get_tensor(image_bytes=image)
        test_loader = DataLoader([(tensor, 0)], 1)
        filename = "keanupoint.jpg"
        for img, label in test_loader:
            pred = model(img)
            break
        probs = F.softmax(pred, dim=1)[0].tolist()
        return render_template(
            "result.html",
            pred=str(round(probs[0] * 100, 2)),
            image=os.path.join(app.config["UPLOAD_FOLDER"], filename),
        )


if __name__ == "__main__":
    model = KeanuModel(3 * 64 * 64, hidden_size=32, out_size=2)
    pth_path = os.path.join(app.config["UPLOAD_FOLDER"], "checkpoint.pth")
    model.load_state_dict(torch.load(pth_path)["state_dict"])
    app.run(debug=True)
