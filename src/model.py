#from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
import pandas as pd
import cv2
import random

# Disables warning, doesn't enable AVX/FMA
import datetime, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#The dataset used in this assignment is a benchmark dataset to use 
import tensorflow as tf
from tensorflow import keras

#Import from keras
from keras.models import Sequential
from keras.layers import Conv2D, GlobalAvgPool2D, MaxPool2D, Flatten, Dense

#Making new prediction
import numpy as np
from keras.preprocessing import image

#Store data
import pickle

#GLOBALE VALUES
img_size = 256
user_color_images = False

CATEGORIES = [
    'dyed-lifted-polyps', 
    'dyed-resection-margins', 
    'esophagitis', 
    'normal-cecum', 
    'normal-pylorus', 
    'normal-z-line',
    'polyps', 
    'ulcerative-colitis'
]

#FUNCTION DEFENITIONS
def create_training_data(): 
    training_data = []   
    for category in CATEGORIES:
        path = os.path.join('data/', category)
        class_num = CATEGORIES.index(category)

        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img)) if user_color_images else cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) 
                img_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([img_array, class_num])
            except Exception:
                print('Building training data for ' + str(category))
            
    random.shuffle(training_data)
    print('Training data is now finnished')
    return training_data

def create_file(name, data):
    pickle_out = open("loaded_data/"+str(name)+".pickle","wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()

def defining_features_and_labels():
    (X, y) = ([], [])

    try:
        pickle_in = open("loaded_data/X.pickle","rb")
        X = pickle.load(pickle_in)

        pickle_in = open("loaded_data/y.pickle","rb")
        y = pickle.load(pickle_in)

    except Exception as e:
        print('Error: '+ str(e))

        training_data = create_training_data()

        for feature, label in training_data:
            X.append(feature)
            y.append(label)

        X = np.array(X).reshape(-1, img_size, img_size, 3 if user_color_images else 1)        
        X = X/255.0

        create_file('y', y)
        create_file('X', X)
        
    return (X, y)

def build_model():
    return Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(img_size, img_size, 3 if user_color_images else 1)),
        MaxPool2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPool2D((2, 2)),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPool2D((2, 2)),
        Flatten(),
        Dense(units=128, activation='relu'),
        Dense(units=8, activation="softmax")
    ])

def plot_history(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

(train_features, train_lables) = defining_features_and_labels()
model = build_model()

checkpoint_path = "trained/model.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    save_weights_only=True, 
    verbose=1, 
    monitor=['val_loss', 'val_accuracy'])

try:
    model.load_weights(checkpoint_path)
except Exception as e:
    print('Exception:' + str(e))
 
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

history = model.fit(
    train_features, 
    train_lables, 
    batch_size=32, 
    epochs=10, 
    validation_split=0.05,
    callbacks=[ cp_callback ]) 

plot_history(history)