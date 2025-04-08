import os
import sys
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image


def load_data(data_path, transform):
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    
    train_size = int(0.8* len(dataset))
    test_size = len(dataset) - train_size
    
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print("Class Labels: ", dataset.classes)

    for images, labels in train_loader:
        print("Labels in this batch:", labels.tolist())  # Print label indices
        print("Class Names:", [dataset.classes[i] for i in labels])  # Convert indices to class names
        break  # Only check one batch
    
    return train_loader, test_loader, dataset.classes