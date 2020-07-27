from flask import Flask, request, render_template
from commons import get_tensor
from inference import getPred
from commons import get_model
import cv2
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

input_size = 3 * 64 * 64
num_classes = 2
hidden_size = 64
hidden_size2 = 32
hidden_size3 = 16


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class KeanuModel(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, hidden_size3)
        self.linear4 = nn.Linear(hidden_size3, num_classes)

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

    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x["val_acc"] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print(
            "Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch, result["val_loss"], result["val_acc"]
            )
        )


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
        pth_path = os.path.join(app.config["UPLOAD_FOLDER"], "checkpoint.pth")
        model = KeanuModel(3 * 64 * 64, hidden_size=32, out_size=2)
        model.load_state_dict(torch.load(pth_path)["state_dict"])
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
