from __future__ import absolute_import, division, print_function, unicode_literals
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import reduce_mean
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.compat.v1.nn import sigmoid_cross_entropy_with_logits as loss
from tensorflow.keras.layers import Dense, BatchNormalization, Conv2DTranspose, Conv2D

image_ids = glob.glob('../input/data/barretts/*')

crop = (30, 55, 150, 175)
images = [np.array((Image.open(i).crop(crop)).resize((64, 64))) for i in image_ids]

for i in range(len(images)):
    images[i] = ((images[i] - images[i].min()) / (255 - images[i].min()))
    images[i] = images[i] * 2 - 1

images = np.array(images)

# hyperparameters
beta1 = 0.5
alpha = 0.2
smooth = 0.9
noise_size = 100
learning_rate = 0.0002
input_shape = (64, 64, 3)

print(tf.__version__)


def generator(noise, reuse=False, alpha=0.2, training=True):
    logits = Sequential([
        Dense((4 * 4 * 512), input_shape=noise, activation='relu'),
        tf.reshape((-1, 4, 4, 512)),
        BatchNormalization(training=training),
        tf.maximum(0.),

        Conv2DTranspose(256, 5, 2, padding='same'),
        BatchNormalization(training=training),
        tf.maximum(0.),

        Conv2DTranspose(128, 5, 2, padding='same'),
        BatchNormalization(training=training),
        tf.maximum(0.),

        Conv2DTranspose(64, 5, 2, padding='same'),
        BatchNormalization(training=training),
        tf.maximum(0.),
        Conv2DTranspose(3, 5, 2, padding='same')

    ], reuse=reuse)
    out = tf.tanh(logits)
    return out, logits


def discriminator(x, reuse=False, alpha=0.2, training=True):
    logits = Sequential([
        Conv2D(x, 32, 5, 2, padding='same'),
        tf.maximum(alpha * x, x),
        Conv2D(x, 64, 5, 2, padding='same'),
        BatchNormalization(x, training=training),
        tf.maximum(alpha * x, x),
        Conv2D(x, 128, 5, 2, padding='same'),
        BatchNormalization(x, training=training),
        tf.maximum(alpha * x, x),
        Conv2D(x, 256, 5, 2, padding='same'),
        BatchNormalization(x, training=training),
        tf.maximum(alpha * x, x),
        tf.reshape(x, (-1, 4 * 4 * 256)),
        Dense(1)
    ], reuse=reuse)
    out = tf.sigmoid(logits)
    return out, logits

'''
def inputs(real_dim, noise_dim):
    inputs_real = tf.compat.v1.placeholder(tf.float32, (None, *real_dim), name='input_real')
    inputs_noise = tf.compat.v1.placeholder(tf.float32, (None, noise_dim), name='input_noise')
    return inputs_real, inputs_noise
'''

# building the graph
# tf.reset_default_graph()

input_real, input_noise = input_shape, noise_size
gen_noise, gen_logits = generator(input_noise)
dis_out_real, dis_logits_real = discriminator(input_real)
dis_out_fake, dis_logits_fake = discriminator(gen_noise, reuse=True)

# defining losses
shape = dis_logits_real

dis_loss_real = reduce_mean(loss(logits=dis_logits_real, labels=tf.ones_like(shape * smooth)))
dis_loss_fake = reduce_mean(loss(logits=dis_logits_fake, labels=tf.zeros_like(shape)))
gen_loss = reduce_mean(loss(logits=dis_logits_fake, labels=tf.ones_like(shape * smooth)))
dis_loss = dis_loss_real + dis_loss_fake

# defining optimizers
total_vars = tf.compat.v1.trainable_variables()

dis_vars = [var for var in total_vars if var.name[0] == 'd']
gen_vars = [var for var in total_vars if var.name[0] == 'g']
dis_opt = Adam(learning_rate=learning_rate, beta1=beta1).minimize(dis_loss, var_list=dis_vars)
gen_opt = Adam(learning_rate=learning_rate, beta1=beta1).minimize(gen_loss, var_list=gen_vars)

batch_size = 128
epochs = 15
iters = len(image_ids) // batch_size
saver = tf.train.Saver(var_list=gen_vars)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for e in range(epochs):

        for i in range(iters - 1):

            batch_images = images[i * batch_size:(i + 1) * batch_size]
            batch_noise = np.random.uniform(-1, 1, size=(batch_size, noise_size))

            sess.run(dis_opt, feed_dict={input_real: batch_images, input_noise: batch_noise})
            sess.run(gen_opt, feed_dict={input_real: batch_images, input_noise: batch_noise})

            if i % 50 == 0:
                print("Epoch {}/{}...".format(e + 1, epochs), "Batch No {}/{}".format(i + 1, iters))

        loss_dis = sess.run(dis_loss, {input_noise: batch_noise, input_real: batch_images})
        loss_gen = gen_loss.eval({input_real: batch_images, input_noise: batch_noise})

        print("Epoch {}/{}...".format(e + 1, epochs), "Discriminator Loss: {:.4f}...".format(loss_dis),
              "Generator Loss: {:.4f}".format(loss_gen))

        # sample_noise = np.random.uniform(-1, 1, size=(8, noise_size))
        # gen_samples = sess.run(generator(input_noise, reuse=True, alpha=alpha), feed_dict={input_noise: sample_noise})
        # view_samples(-1, gen_samples, 2, 4, (10, 5))
        # plt.show()

        saver.save(sess, './checkpoints/generator.ckpt')
