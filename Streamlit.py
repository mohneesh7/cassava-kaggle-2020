import os
import glob
import json
import numpy as np
import tensorflow.keras as k
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow.keras as k
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K 
import efficientnet.tfkeras as efn
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications import EfficientNetB0
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)


IMG_SIZE = 512
size = (IMG_SIZE,IMG_SIZE)
best_model = k.models.load_model('https://github.com/mohneesh7/cassava-kaggle-2020/blob/main/models/Cassava_best_model_effnetb4.h5',compile=False)
print(best_model)
TEST_DIR = 'D:\\Projects\\Redundant Stuff\\Cassava Leaf Disease Detection\\test_images\\'
test_images = os.listdir(TEST_DIR)
predictions = []

for image in test_images:
    img = Image.open(TEST_DIR + image)
    img = img.resize(size)
    print(img)
    img = np.expand_dims(img, axis=0)
    print(img.shape)
    predictions.extend(best_model.predict(img).argmax(axis = 1))
    print(best_model.predict(img))
