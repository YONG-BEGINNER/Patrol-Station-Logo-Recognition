import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from CNN_Model import SimpleCNN
from torchvision import transforms
from preprocessing_pytorch_method import load_data
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

data_path = "./Crop"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current Device: {device}")
print(os.path.abspath(data_path))

# Set the preprocessing step that need to apply to the images
transform = transforms.Compose([
    transforms.Resize((50,50)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Call the function to load the data into the variable
train_data, test_data, classes = load_data(data_path, transform)


num_classes = len(classes)
model = SimpleCNN(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define Epoch that want to be perform
for epoch in range(5):
    # Set the model to train state
    model.train()

    # Initialize the performance tracking variable
    running_loss = 0.0
    correct = 0
    total = 0

    # Loop through the images and labels
    for images, labels in train_data:
        # Assign to the device that use to perform training
        images, lables = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Reset the gradient after each batch size, so that the gradient doens't accumulate
        optimizer.zero_grad()
        # Backpropagation
        loss.backward()
        # Update model weight base on gradient
        optimizer.step()

        # running_loss = running_loss + loss.item()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        # Sum up the correct labels and convert it from tensor into python int
        # predicted = [1, 2, 0] labels=[1, 0, 0] -> [True, False, True].sum() = 2
        correct +=(predicted == labels).sum().item()
        total += labels.size(0)

    print (f"Epoch {epoch+1}, Loss:{running_loss:.3f}, Accuracy:{100*correct/total:.2f}%") 

model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_data:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _,predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

plt.figure(figsize=(8,6))
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
