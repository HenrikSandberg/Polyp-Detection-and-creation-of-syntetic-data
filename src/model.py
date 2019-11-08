import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
import cv2
import random
from sklearn import metrics
from sklearn import svm

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

        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img)) if USE_COLOR else cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) 
                img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([img_array, class_num])
            except Exception:
                print('Building training data for ' + str(category))
            
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
def defining_features_and_labels():
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
        X = X/255.0


    #Splits and sets aside data for validation of the models performasce
    trainging_size = 0.95
    X_split = int(len(X)*trainging_size)
    X1 = X[0: X_split]
    X2 = X[X_split: ]

    y_split = int(len(y)*trainging_size)
    y1 = y[0: y_split]
    y2 = y[y_split: ]
    
    create_file('y', y)
    create_file('X', X)

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
        Dense(units=8, activation="softmax")
    ])
#Dense(units=128, activation='relu'),

# Plots the distribution between accuracy and validation accuracy
def plot_history(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

(train_features, train_lables), (x_test, y_test) = defining_features_and_labels()
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
'''
history = model.fit(
    train_features, 
    train_lables, 
    batch_size=32, 
    epochs=15, 
    validation_split=0.15,
    callbacks=[ cp_callback, tensorboard ]) 
'''


y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis = 1)

for num in range(len(y_pred)):
    print('[PRED][ACTUAL] = [' + CATEGORIES[y_pred[num]]+'] ['+ CATEGORIES[y_test[num]]+']')


score = metrics.classification_report(y_test, y_pred)
print(score)