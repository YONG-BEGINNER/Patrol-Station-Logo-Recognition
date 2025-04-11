import os
import sys
sys.path.append("./../pytorch")
import torch
from PIL import Image
from torchvision import transforms
from CNN_Model import SimpleCNN
from pytorch_beginner import num_classes_return
from flask import Flask, render_template, request, jsonify

data_path = "./../Crop"

app = Flask(__name__)
num_class = num_classes_return(data_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_file = []
for files in os.listdir('./'):
    if files.endswith('.pth'):
        model_file.append(files)
print(model_file[0])

model = SimpleCNN(num_classes = num_class)
model.load_state_dict(torch.load(model_file[0], map_location=device))
model.eval()

transform = transforms.Compose({
    transforms.Resize((50,50)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
})

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        img_file = request.files['images']
        img = Image.open(img_file.stream)

        # Apply transformation
        img = transform(img).unsqueeze(0).to(device)

        # Make Prediction
        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            prediction = predicted.item()

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
