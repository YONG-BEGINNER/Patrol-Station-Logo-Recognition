import os
import cv2
import math
import numpy as np
from preprocessing import load_img, train_test_split_func

data_path = "./Patrol Recognition/Crop"
print(os.path.abspath(data_path))
target_size = (50,50)

x, y = load_img(data_path, target_size)
x_train, x_test, y_train, y_test = train_test_split_func(x, y, 0.2, 42)

