import os
import torch
from torch.utils.data import dataset
from torchvision import transforms
from PIL import Image

class CustomImageDataset(dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.img_path = []
        self.labels = []
        
        for idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir,class_name)
            for img_name in os.listdir(class_dir):
                self.img_path.append(os.path.join(class_dir, img_name))