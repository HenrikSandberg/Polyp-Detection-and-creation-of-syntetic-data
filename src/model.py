import numpy as np
import time

from sklearn import metrics
from sklearn import svm

#Data preprocessing
import pickle
import cv2
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
Genaerate trainingdata by moving in to the different directories, using the name on the directories as labels for the data. 
The data from the images are together withe the labels added into an array. 
At the end, the data is suffeld in order to make sure that alle the calssifications are moved around. '''
def create_training_data(): 
    training_data = []   
    for category in CATEGORIES:
        path = os.path.join('data/', category)
        class_num = CATEGORIES.index(category)

        print('Building training data for ' + str(category))
        for img in os.listdir(path):
            try:
                #y=25
                #x=50
                #h=400
                #w=500
                
                img_path = os.path.join(path,img)
                img_file = cv2.imread(img_path) if USE_COLOR else cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                img_array = cv2.resize(img_file, (IMG_SIZE, IMG_SIZE))
                training_data.append([img_array, class_num])

                #crop = img_file[y:y+h, x:x+w]
                #cv2.imshow(img_array, crop)
                #cv2.waitKey(0)
                #plt.imshow(img_array)
                #plt.show()                
            except Exception:
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
        Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS)),
        MaxPool2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPool2D((2, 2)),
        Conv2D(256, (3, 3), activation='relu'),
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

(features, labels) = create_features_and_labels()
(x_train, y_train), (x_test, y_test) = split_into_train_and_test(features, labels)
model = build_model()

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

checkpoint_path = "training_checkpoints_CNN/"+NAME+".h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    save_weights_only=True, 
    verbose=1, 
    monitor='val_loss')

try:
    model.load_weights(checkpoint_path)
except Exception as e:
    print('Exception:' + str(e))
 
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['sparse_categorical_accuracy', 'accuracy'])

history = model.fit(
    x_train, 
    y_train, 
    batch_size=32, 
    epochs=7, 
    validation_split=0.15,
    callbacks=[ cp_callback, tensorboard ]) 

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis = 1)

#for num in range(len(y_pred)):
#    print('[PRED][ACTUAL] = [' + CATEGORIES[y_pred[num]]+'] ['+ CATEGORIES[y_test[num]]+']')

score = metrics.classification_report(y_test, y_pred)
print(score)

'''
WITH COLOR
             precision    recall  f1-score   support

           0       0.71      0.56      0.63        48
           1       0.67      0.78      0.72        51
           2       0.60      0.85      0.70        47
           3       0.85      0.92      0.88        37
           4       0.97      0.92      0.94        62
           5       0.83      0.59      0.69        59
           6       0.75      0.66      0.70        41
           7       0.78      0.82      0.80        55

    accuracy                           0.76       400
   macro avg       0.77      0.76      0.76       400
weighted avg       0.78      0.76      0.76       400


             precision    recall  f1-score   support

           0       0.76      0.79      0.78        24
           1       0.76      0.73      0.74        22
           2       0.90      0.41      0.56        22
           3       0.95      0.90      0.92        20
           4       0.76      1.00      0.87        13
           5       0.63      0.88      0.73        25
           6       0.81      0.68      0.74        19
           7       0.71      0.80      0.75        15

    accuracy                           0.76       160
   macro avg       0.79      0.77      0.76       160
weighted avg       0.78      0.76      0.75       160


WITOUT COLOR
              precision    recall  f1-score   support

           0       0.61      0.55      0.58        55
           1       0.68      0.67      0.67        54
           2       0.68      0.73      0.70        52
           3       0.75      0.82      0.79        40
           4       0.88      0.80      0.84        56
           5       0.64      0.64      0.64        44
           6       0.64      0.38      0.47        56
           7       0.49      0.79      0.60        43

    accuracy                           0.66       400
   macro avg       0.67      0.67      0.66       400
weighted avg       0.67      0.66      0.66       400
'''