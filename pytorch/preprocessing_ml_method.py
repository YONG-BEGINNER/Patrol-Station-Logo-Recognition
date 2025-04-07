import os
import cv2
import sys
import math
import numpy as np
from sklearn.model_selection import train_test_split

def load_imgs(data_path, target_size):
    images = []
    labels = []
    
    for dir in os.listdir(data_path):
        dir_name = os.path.join(data_path,dir)
        for file in os.listdir(dir_name):
            img_path = os.path.join(dir_name, file)
            print(img_path.replace("\\","/"))
            if file[-3:] in (".jpg", ".png"):
                img = cv2.imread(img_path.replace("\\","/"))
                img = cv2.resize(img, target_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                labels.append(dir)
    images = np.array(images)
    labels = np.array(labels)
    
    return images,labels

def train_test_split_func(x, y, size, random):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = size, random_state = random)
    return x_train, x_test, y_train, y_test
