import os
import cv2
import math
import torch
import numpy as np
from preprocessing_pytorch_method import load_data
from CNN_Model import SimpleCNN
from preprocessing_ml_method import train_test_split_func
from torchvision import transforms, datasets
import torch.optim as optim
import torch.nn as nn

data_path = "./Crop"
print(os.path.abspath(data_path))

transform = transforms.Compose([
    transforms.Resize((50,50)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_data, test_data, classes = load_data(data_path, transform)

num_classes = len(classes)
model = SimpleCNN(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_data:
        output = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct +=(predicted == labels).sum().item()
        total += labels.size(0)

    print (f"Epoch {epoch+1}, Loss:{running_loss:.3f}, Accuracy:{100*correct/total:.2f}%") 





