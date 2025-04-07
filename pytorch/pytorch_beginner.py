import os
import cv2
import math
import numpy as np
from preprocessing_pytorch_method import load_data
from preprocessing_ml_method import train_test_split_func
from torchvision import transforms, datasets

data_path = "./Crop"
print(os.path.abspath(data_path))

transform = transforms.Compose([
    transforms.Resize((50,50)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

x, y = load_data(data_path, transform)
x_train, x_test, y_train, y_test = train_test_split_func(x, y, 0.2, 42)



