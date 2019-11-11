import numpy as np
import time

from sklearn import metrics
from sklearn import svm

#tensorboard --logdir='logs/fit/'

#Data preprocessing
import pickle
import cv2
import imutils
import random
import matplotlib.pyplot as plt

# Disables warning, doesn't enable AVX/FMA
import datetime, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#The dataset used in this assignment is a benchmark dataset to use 
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Conv2D, GlobalAvgPool2D, MaxPool2D, Flatten, Dense

#GLOBALE VALUES
IMG_SIZE = 64
USE_COLOR = True
CHANNELS = 3 if USE_COLOR else 1
COLOR_STATE = "color" if USE_COLOR else "gray"

NAME = "model_{}_{}".format(IMG_SIZE, COLOR_STATE)

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

'''
Looks trough the image for a green box, if found it will create a black box over
that hids what ever is inside the box. This is a precorsen to make sure the model
learns just to look inside the box for information.
'''
def remove_green_box(img_file, img_path, img):
    out_file = img_file if USE_COLOR else cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    hsv = cv2.cvtColor(img_file, cv2.COLOR_BGR2HSV)

    lower_green = np.array([17,163,134])
    upper_green = np.array([105, 240, 197])

    mask = cv2.inRange (hsv, lower_green, upper_green)
    green_content = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    if len(green_content) > 0:
        green_area = max(green_content, key = cv2.contourArea)
        (xg,yg,wg,hg) = cv2.boundingRect(green_area)

        if hg*wg > 35000 and xg < 50 and yg > 200:
            v1 = xg+wg if wg < 250 else 209+38
            v2 = yg+hg if hg < 200 else 168+384
            yg = yg if yg > 350 else 384
            cv2.rectangle(out_file,(xg,yg),(v1, v2),(0,0,0), cv2.FILLED)

    return out_file

''' 
Genaerate trainingdata by moving in to the different directories, using the name on the directories as labels for the data. 
The data from the images are together withe the labels added into an array. 
At the end, the data is suffeld in order to make sure that alle the calssifications are moved around. 
'''
def create_training_data(): 
    training_data = []   
    for category in CATEGORIES:
        path = os.path.join('data/', category)
        class_num = CATEGORIES.index(category)

        print('Building training data for ' + str(category))
        for img in os.listdir(path):
            try:                
                img_path = os.path.join(path,img)
                img_file = cv2.imread(img_path) 
                out_file = remove_green_box(img_file, img_path, img)
                img_array = cv2.resize(out_file, (IMG_SIZE, IMG_SIZE))
                training_data.append([img_array, class_num])    

            except Exception as e:
                print(e)
                pass
            
    random.shuffle(training_data)
    return training_data

# Creates an pickle fil. This can directly be implemented into the model for quicker training
def create_file(name, data):
    pickle_out = open("loaded_data/"+name+"_"+str(IMG_SIZE)+"_"+COLOR_STATE+".pickle","wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()

''' 
This function attemts to load the pickle files into memory.
If the two pickle files do not exist, the function will then 
use the two privious functions to create the training data and 
save them into files.
'''
def create_features_and_labels():
    (X, y) = ([], [])

    try:
        pickle_in = open("loaded_data/X_"+str(IMG_SIZE)+"_"+str(COLOR_STATE)+".pickle","rb")
        X = pickle.load(pickle_in)

        pickle_in = open("loaded_data/y_"+str(IMG_SIZE)+"_"+str(COLOR_STATE)+".pickle","rb")
        y = pickle.load(pickle_in)

    except Exception as e:
        print('Error: '+ str(e))

        training_data = create_training_data()

        for feature, label in training_data:
            X.append(feature)
            y.append(label)

        X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, CHANNELS)        
        X = (X-127.0)/127.0
        #X = X / 255.0
        
        create_file('y', y)
        create_file('X', X)

    #Splits and sets aside data for validation of the models performasce
    return (X,y)


def split_into_train_and_test(X, y, trainging_size = 0.98):
    X_split = int(len(X)*trainging_size)
    X1 = X[0: X_split]
    X2 = X[X_split: ]

    y_split = int(len(y)*trainging_size)
    y1 = y[0: y_split]
    y2 = y[y_split: ]

    return (X1, y1), (X2, y2)

''' 
Builds a CNN model with two hidden layers. The model uses a softmax in order to determen which 
category is the most correct. 
'''
def build_model():
    return Sequential([
        Conv2D(64, (2, 2), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS)),
        MaxPool2D((2, 2)),
        Conv2D(128, (2, 2), activation='relu'),
        MaxPool2D((2, 2)),
        Conv2D(256, (2, 2), activation='relu'),
        MaxPool2D((2, 2)),
        Flatten(),
        Dense(units=128, activation='relu'),
        Dense(units=8, activation="softmax")
    ])

# Plots the distribution between accuracy and validation accuracy
def plot_history(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

#Creating model, data and splits training data from validation data
(features, labels) = create_features_and_labels()
(x_train, y_train), (x_test, y_test) = split_into_train_and_test(features, labels)
model = build_model()

#Generate log to Tensorbord
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + NAME
tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

#Generate a trained model
checkpoint_path = "training_checkpoints_CNN/"+NAME+".h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    save_weights_only=True, 
    verbose=1, 
    monitor='val_loss')

#Load the pretraind model if it exist
try:
    model.load_weights(checkpoint_path)
except Exception as e:
    print('Exception:' + str(e))

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

history = model.fit(
    x_train, 
    y_train, 
    batch_size=32, 
    epochs=7, 
    validation_split=0.15,
    callbacks=[ cp_callback, tensorboard ]) 

#Evaluate the models performance
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis = 1)

score = metrics.classification_report(y_test, y_pred)
print(score)