import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

# class CustomImageDataset(dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.classes = os.listdir(root_dir)
#         self.img_path = []
#         self.labels = []
        
#         for idx, class_name in enumerate(self.classes):
#             class_dir = os.path.join(root_dir,class_name)
#             for img_name in os.listdir(class_dir):
#                 self.img_path.append(os.path.join(class_dir, img_name))

transform = transforms.Compose([
    transforms.Resize((50,50)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

data_path = "./Crop"
dataset = datasets.ImageFolder(root=data_path, transform=transform)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

print("Class Labels: ", dataset.classes)
print("Sample Image Shape: ", dataset[0][0].shape)

for images, labels in dataloader:
    print("Labels in this batch:", labels.tolist())  # Print label indices
    print("Class Names:", [dataset.classes[i] for i in labels])  # Convert indices to class names
    break  # Only check one batch