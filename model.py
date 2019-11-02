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
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense

#Fitting the CNN to images
from keras.preprocessing.image import ImageDataGenerator

#Making new prediction
import numpy as np
from keras.preprocessing import image

#Store data
import pickle

#GLOBALE VALUES
img_size = 256
data_dir = 'data/'

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
                img_array = cv2.imread(os.path.join(path,img)) #TODO: May want to make img gray -> can add: , cv2.IMREAD_GRAYSCALE
                new_img_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([new_img_array, class_num])
                # plt.imshow(img_array, cmap='gray')
                # plt.show()
                # break
            except Exception as e:
                print(e)
    random.shuffle(training_data)
    return training_data

def build_model():
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
        MaxPool2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPool2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(8, activation='softmax')
    ])

    base_model = keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3)
    )

    golbal_avg_pool = keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = Dense(8, activation="softmax")(golbal_avg_pool)
    model = Model(inputs=base_model.input, outputs=output)
    model.summary()
    return model

X = []
y = []

''' try:
    pickle_in = open("X.pickle","rb")
    X = pickle.load(pickle_in)
    pickle_in = open("y.pickle","rb")
    y = pickle.load(pickle_in)
    # X = X/255.0
except Exception as e: '''
training_data = create_training_data()

for feature, label in training_data:
    X.append(feature)
    y.append(label)

X = np.array(X).reshape(-1, img_size, img_size, 3)
X = X/255

''' pickle_out = open('X.pickle', 'wb')
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open('y.pickle', 'wb')
pickle.dump(y, pickle_out)
pickle_out.close() '''

model = build_model()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=[ 'accuracy' ]
)

#Add a callback so that we can use tensorboard
logdir = os.path.join("logs_l8", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

history = model.fit(X,y,
    batch_size=32,
    epochs=10, 
    validation_split=0.15,
)

''' callbacks=[
    tensorboard_callback,
    keras.callbacks.Model.Checkpoint('model.h5', monitor='val_loss')
]
model = keras.models.load_data('model.h5')
 '''