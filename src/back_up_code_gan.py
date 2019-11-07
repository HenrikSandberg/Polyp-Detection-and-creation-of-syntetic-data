from keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import cv2
import random
import tensorflow as tf

r=0
img_size = 256

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

def create_training_data(selected=0): 
    training_data = []   
    category = CATEGORIES[selected]
    path = os.path.join('data/', category)

    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path,img))
            new_img_array = cv2.resize(img_array, (img_size, img_size))
            training_data.append([new_img_array])
        except Exception:
            print('Building training data for ' + str(category))
            
    random.shuffle(training_data)

    return np.array(training_data).reshape(-1, img_size, img_size, 3) / 255.0

def init_weights(shape):
    return tf.Variable(tf.random.normal(shape, stddev=0.1))

def init_bias(shape):
    return tf.Variable(tf.constant(0.2, shape=shape))

class Generator:
    def __init__(self):
        with tf.compat.v1.variable_scope('g'):
            self.gW1 = init_weights([100, 256])
            self.gb1 = init_bias(256)
            self.gW2 = init_weights([256, 784])
            self.gb2 = init_bias([784])

    def forward(self, z, training= True):
        fc1 = tf.matmul(z, self.gW1) + self.gb1
        fc1 = tf.layers.batch_normalization(fc1, training=training)
        fc1 = tf.nn.leaky_relu(fc1)

        #The sigmoid nomralizes the data (value between 0-1)
        fc2 = tf.nn.sigmoid(tf.matmul(fc1, self.gW2)+ self.gb2)
        return fc2

class Discriminator:
    def __init__(self):
        with tf.compat.v1.variable_scope('d'):
            self.dW1 = init_weights([5, 5 ,1, 16])
            self.db1 = init_bias([16])
            self.dW2 = init_weights([3, 3, 16, 32])
            self.db2 = init_bias([32])

            self.W3 = init_weights([1568, 128])
            self.b3 = init_bias([128])
            self.W4 = init_weights([128, 1])
            self.b4 = init_bias([1])

    def forward(self, X):
        self.X = tf.reshape(X, shape=[-1, 28, 28, 1])
        conv1 = tf.nn.leaky_relu(tf.nn.conv2d(self.X, self.dW1, strides=[1, 2, 2, 1], padding='SAME')+ self.db1)
        conv1 = tf.layers.batch_normalization(conv1, True)
        conv2 = tf.nn.leaky_relu(tf.nn.conv2d(conv1, self.dW2, strides=[1,2,2,1], padding='SAME') + self.db2)
        conv2 = tf.layers.batch_normalization(conv2, True)
        conv2 = tf.reshape(conv2, shape=[-1, 7*7*32])

        fc1 = tf.nn.leaky_relu(tf.matmul(conv2, self.W3)+self.b3)
        logits = tf.matmul(fc1, self.W4) + self.b4
        fc2 = tf.nn.sigmoid(logits)

        return fc2, logits


def cost(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

def next_batch(data, size):
    global r
    if r*size+size > len(data):
        r = 0
    x_train_batch = data[size*r:r*size+size, :]
    r = r + 1
    return x_train_batch



# data = create_training_data()
d = Discriminator()
g= Generator()

phX = tf.compat.v1.placeholder(tf.float32, [None, 784])
phZ = tf.compat.v1.placeholder(tf.float32, [None, 100])

G_out = g.forward(feed_dict={phZ})
G_out_sample = g.forward(feed_dict={phZ}, training=False)

D_out_real, D_logits_real = d.forward(feed_dict={phX})
D_out_fake, D_logits_fake = d.forward(G_out)

D_real_loss = cost(D_logits_real, tf.ones_like(D_logits_real))
D_fake_loss = cost(D_logits_fake, tf.zeros_like(D_logits_fake))
D_loss = D_real_loss + D_fake_loss

G_loss = cost(D_logits_fake, tf.ones_like(D_logits_fake))

#Init learning rate
lr = 0.001

#Epochs per Lable
epochs = 7000

# Predictiong epochs out of total epochs [After predicting images are generated]
prediction_epochs = 4000
batch_size = 50

train_vars = tf.trainable_variables()