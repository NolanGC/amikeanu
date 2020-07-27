import os
import cv2
import torch
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from flask import Flask, request, render_template

def get_tensor(image_bytes):
    my_transforms = transforms.ToTensor()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    open_cv_image = np.array(pil_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    new_array = cv2.resize(open_cv_image, (64, 64))
    img_tensor = torch.tensor(new_array, dtype=torch.float)
    return img_tensor

class KeanuModel(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super().__init__()
        self.linear1 = nn.Linear(3 * 64 * 64, 32)
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, 2)

    def forward(self, xb):
        # Flatten the image tensors
        xb = xb.view(xb.size(0), -1)
        # Get intermediate outputs using hidden layer
        out = self.linear1(xb)
        # Apply activation function
        out = F.relu(out)
        # Get predictions using output layer
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = self.linear4(out)
        return out

app = Flask(__name__) 
app.config["UPLOAD_FOLDER"] = os.path.join("static", "images")

model = KeanuModel(3 * 64 * 64, hidden_size=32, out_size=2)
        pth_path = os.path.join(app.config["UPLOAD_FOLDER"], "checkpoint.pth")
        model.load_state_dict(torch.load(pth_path)["state_dict"])

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
    app.run(debug=True)
