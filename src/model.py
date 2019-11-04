from __future__ import absolute_import, division, print_function, unicode_literals

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

data_dir = 'data/'

train_data = []
train_lable = []

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
        path = os.path.join(data_dir, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) #TODO: May want to make img gray -> can add: 
                new_img_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([new_img_array, class_num])
                # plt.imshow(img_array, cmap='gray')
                # plt.show()
                # break
            except Exception as e:
                print(e)
    random.shuffle(training_data)
    return training_data

def define_data():
    (X, y) = ([], [])

    try:
        pickle_in = open("X.pickle","rb")
        X = pickle.load(pickle_in)

        pickle_in = open("y.pickle","rb")
        y = pickle.load(pickle_in)
    except Exception as e:
        print('Error accured: '+ str(e))

        training_data = create_training_data()

        for feature, label in training_data:
            X.append(feature)
            y.append(label)

        X = np.array(X).reshape(-1, img_size, img_size, 1)
        X = X/255.0
        
        pickle_out = open("X.pickle","wb")
        pickle.dump(X, pickle_out)
        pickle_out.close()

        pickle_out = open("y.pickle","wb")
        pickle.dump(y, pickle_out)
        pickle_out.close()
    return (X, y)

def build_model():
    model = Sequential([
        Conv2D(256, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
        MaxPool2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPool2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPool2D((2, 2)),
        Flatten(),
        Dense(units=128, activation='relu'),
        Dense(units=8, activation="softmax")
    ])
    return model

(train_data, train_lable) = define_data()
model = build_model()

#Add a callback so that we can use tensorboard
logdir = os.path.join("logs_l8", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

history = model.fit(
    train_data, 
    train_lable, 
    batch_size=32, 
    epochs=10, 
    validation_split=0.05,
    callbacks=[
        tensorboard_callback,
        keras.callbacks.ModelCheckpoint(
        filepath='mymodel_{epoch}.h5',
        # Path where to save the model
        # The two parameters below mean that we will overwrite
        # the current checkpoint if and only if
        # the `val_loss` score has improved.
        save_best_only=True,
        monitor='val_loss',
        verbose=1)
    ]) 

model = keras.models.load_data('model.h5')

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()