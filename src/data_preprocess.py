import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint

np.random.seed(42)

data_dir = 'path/to/dataset'

labels = ['dog', 'not_dog']

dataset = []

for label in labels:
    label_dir = os.path.join(data_dir, label)
    for filename in os.listdir(label_dir):
        image_path = os.path.join(label_dir, filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (128, 128))
        dataset.append((image, label))
        
        #skip blurry images edge case
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur < 100:
            # If the image is too blurry, skip it
            print("Image is too blurry. Skipping...")
        else:
            # If the image is not too blurry, use it
            # Preprocess the image by resizing and normalizing
            image = cv2.resize(image, (64, 64))
            image = image / 255.0
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (128, 128))
        dataset.append((image, label))

