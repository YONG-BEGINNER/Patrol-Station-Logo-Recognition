import os
import sys
sys.path.append("./../pytorch")
import uuid
import torch
import threading
from PIL import Image
from torchvision import transforms
from Pytorch.CNN_Model import SimpleCNN
from Pytorch.pytorch_beginner import num_classes_return
from flask import Flask, render_template, request, jsonify, after_this_request

# Define The source of the data to get the labeling
data_path = "./../Crop"

# Directory to handle the uploaded file
Upload_Folder = 'static/upload'
os.makedirs(Upload_Folder, exist_ok=True)

app = Flask(__name__)

# Define class label
num_class, classes_label = num_classes_return(data_path)

# Select Model use device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Select the latest trained model
model_file = []
for files in os.listdir('./'):
    if files.endswith('.pth'):
        model_file.append(files)
print(model_file[0])

# Initialize the Model architecture
model = SimpleCNN(num_classes = num_class)
# Load the pretrain model weight
model.load_state_dict(torch.load(model_file[0], map_location=device))
# Set the model to evaluation mode
model.eval()

# Set the preprocessing that need to be used in the uploaded Images
transform = transforms.Compose({
    transforms.Resize((50,50)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
})

# Delete Local Temporary file after user upload to the website.
def delete_image_later(image_path, delay):
    """Deletes image after a delay (in seconds)."""
    threading.Timer(delay, os.remove, args=[image_path]).start()

# Define the route and method assigned to it
@app.route("/", methods=["GET", "POST"])
def index():
    # Initialize Variable
    prediction = None
    filename = None
    image_path = None
    # When method is POST
    if request.method == "POST":
        # Get file from the form element named image
        img_file = request.files['image']

        #Preprocessing Start
        img = Image.open(img_file.stream).convert("RGB")
        
        # Save image with a unique filename, for display purpose
        filename = f"{uuid.uuid4().hex}.png"
        image_path = os.path.join(Upload_Folder, filename)
        img.save(image_path)

        # Apply transformation
        img = transform(img).unsqueeze(0).to(device)

        # Note: Cannot save image after apply transformation, since it is converted to tensor.

        # Make Prediction
        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            prediction = classes_label[predicted.item()]

        # Call the function to delete the image after pridiction finished
        delete_image_later(image_path, delay=2)
            
        return render_template('index2.html',prediction=prediction, image_file=filename)

    return render_template('index2.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
