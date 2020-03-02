'''
My code has used this code created by tensorflow quite extencivly in order to make a functional Generative Adversarial Network. 
The code is different esspecialy in the import of data part of the algorithem. I also hade to get a basic understaning of the
alorithem in order to make it perform as expected. 

Sorce: https://www.tensorflow.org/tutorials/generative/dcgan
'''
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Reshape, Dense, Dropout, Conv2D, Conv2DTranspose, \
    Flatten

import datetime, os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
import numpy as np
import time
import cv2

from IPython import display

# Global variables
IMG_SIZE = 64
CHANNELS = 3

EPOCHS = 1000

NOISE_DIM = 64
num_examples_to_generate = 10
SEED = tf.random.normal([num_examples_to_generate, NOISE_DIM])

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

BUFFER_SIZE = 60000
BATCH_SIZE = 256

DILATION_RATE = (5, 5)

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
SELECTED_CATEGORY = CATEGORIES[0]

train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
'''
Imports a specific category, then splits the content into features and lables which can be used.
It also reshapes the features and normalizes them.
'''


def create_training_data(selected=0):
    training_data = []
    SELECTED_CATEGORY = CATEGORIES[selected]
    path = os.path.join('data/', SELECTED_CATEGORY)

    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img)) if (CHANNELS == 3) else cv2.imread(os.path.join(path, img),
                                                                                               cv2.IMREAD_GRAYSCALE)
            new_img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_img_array, selected])
        except Exception:
            print('Building training data for ' + str(SELECTED_CATEGORY))

    (X, y) = ([], [])

    for feature, label in training_data:
        X.append(feature)
        y.append(label)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, CHANNELS)
    X = (X - 127.5) / 127.5
    return (X, y)


'''
This creates the generator model used in the full GAN process. 
It takes in a nois image and trough the different layers create 
somthing that resebles a specific category. 
'''


def build_generator_model():
    size = int(IMG_SIZE / 4)
    return Sequential([
        Dense(size * size * 256, use_bias=False, input_shape=(NOISE_DIM,)),
        BatchNormalization(),
        LeakyReLU(),
        Reshape((size, size, 256)),
        Conv2DTranspose(128, DILATION_RATE, strides=(1, 1), padding='same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(),
        Conv2DTranspose(128, DILATION_RATE, strides=(1, 1), padding='same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(),
        Conv2DTranspose(64, DILATION_RATE, strides=(2, 2), padding='same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(),
        Conv2DTranspose(CHANNELS, DILATION_RATE, strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])


'''
Generates a discriminator used to detect the quality of the generated image.
'''


def build_discriminator_model():
    return Sequential([
        Conv2D(IMG_SIZE, DILATION_RATE, strides=(2, 2), padding='same', input_shape=[IMG_SIZE, IMG_SIZE, CHANNELS]),
        LeakyReLU(),
        Dropout(0.3),
        Conv2D(128, DILATION_RATE, strides=(2, 2), padding='same'),
        LeakyReLU(),
        Dropout(0.3),
        Conv2D(256, DILATION_RATE, strides=(2, 2), padding='same'),
        LeakyReLU(),
        Dropout(0.3),
        Flatten(),
        Dense(1)
    ])


# Calculate discriminator loss
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


# Calculate generator loss
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


'''
This function takes in an images and uses it as well as a genarated image to test both the generator and the
discriminator. At the end of the function both models are evaluated on ther performance and the result will
be used to ajust their wieghts. 
'''


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


# Activate the training function and forces the models to train for a specific number of epochs
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, SEED)

        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, int(time.time() - start)))

    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, SEED)


# Outputs an image to train folder. This is just used for debugging
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    plt.imshow((predictions[0, :, :, 0] + 127.5) * 127.5)
    plt.axis('off')
    plt.savefig('syntetic/train/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()


'''
Generate synthetic data which can be used for training in the future.
'''


def create_synthetic_data(checkpoint, model):
    checkpoint_dir = './training_checkpoints_GAN/{}/'.format(SELECTED_CATEGORY)

    try:
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        for y in range(0, 100):
            input = tf.random.normal([num_examples_to_generate, NOISE_DIM])
            predictions = model(input, training=False)

            for i in range(predictions.shape[0]):
                # plt.imshow(predictions[i, :, :, -1] ) # 0]+ 127.5) * 127.5
                # img = cv2.cvtColor(, cv2.COLOR_BGR2RGB)
                plt.imshow(predictions[i, :, :, -1])
                plt.axis('off')
                plt.savefig('syntetic/{}/{}_{}_{}.png'.format(SELECTED_CATEGORY, SELECTED_CATEGORY, y, i))
                plt.close()
    except Exception as e:
        print(e)


'''
This loop is the beginning of the running code. It goes through category by category,
trains the algorithems and then generat syntetic data. Then repete the prosess for the
next categorie. 
'''
# for i in range(len(CATEGORIES)):
#    SELECTED_CATEGORY = CATEGORIES[i]
print("Now traingin for category {}".format(SELECTED_CATEGORY))

(train_images, train_labels) = create_training_data(0)
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

generator = build_generator_model()
discriminator = build_discriminator_model()
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints_GAN/{}/'.format(SELECTED_CATEGORY)
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

# train(train_dataset, EPOCHS)
print("Now generating images")
create_synthetic_data(checkpoint, generator)
