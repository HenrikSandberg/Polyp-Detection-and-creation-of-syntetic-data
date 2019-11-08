from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Reshape, Dense, Dropout, Conv2D, Conv2DTranspose, Flatten

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
import numpy as np
import time
import cv2

from IPython import display

IMG_SIZE = 128
CHANNELS = 1

EPOCHS = 5000

NOISE_DIM = IMG_SIZE
num_examples_to_generate = 16
SEED = tf.random.normal([num_examples_to_generate, NOISE_DIM])

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

BUFFER_SIZE = 60000
BATCH_SIZE = 256

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
            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
            new_img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_img_array, selected])
        except Exception:
            print('Building training data for ' + str(category))
    
    (X, y) = ([], [])

    for feature, label in training_data:
        X.append(feature)
        y.append(label)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, CHANNELS)
    return (X, y)

def build_generator_model():
    size = int(IMG_SIZE/4)
    return Sequential([
        Dense(size*size*256, use_bias=False, input_shape=(IMG_SIZE,)),
        BatchNormalization(),
        LeakyReLU(),
        Reshape((size, size, 256)),
        Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(),
        Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(),
        Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(),
        Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])

def build_discriminator_model():
    return Sequential([
        Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[IMG_SIZE, IMG_SIZE, CHANNELS]),
        LeakyReLU(),
        Dropout(0.3),
        Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        LeakyReLU(),
        Dropout(0.3),
        Conv2D(256, (5, 5), strides=(2, 2), padding='same'),
        LeakyReLU(),
        Dropout(0.3),
        Flatten(),
        Dense(1)
    ])

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, SEED)

        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, int(time.time()-start)))
        
    display.clear_output(wait=True)
    generate_and_save_images(generator,epochs,SEED)


def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    #fig = plt.figure(figsize=(4,4))
    
    # for i in range(predictions.shape[0]):
    #     plt.subplot(4, 4, i+1)
    #     plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
    #     plt.axis('off')

    plt.imshow(predictions[0, :, :, 0] * IMG_SIZE + IMG_SIZE, cmap='gray')
    plt.axis('off')
    plt.savefig('syntetic/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()
    # plt.show()

(train_images, train_labels) = create_training_data()
train_images = (train_images - 127.5) / 127.5 
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

generator = build_generator_model()
discriminator = build_discriminator_model()
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints_GAN'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer, 
    discriminator_optimizer=discriminator_optimizer, 
    generator=generator, 
    discriminator=discriminator)

try:
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
except Exception as e:
    print(e)

train(train_dataset, EPOCHS)