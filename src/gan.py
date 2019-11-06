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

GENERATE_RES = 2
DATA_FOLDER = './tf_data/VGAN/MNIST'
img_size = 128 * GENERATE_RES
NOISE_SIZE = 500
BATCH_SIZE = 100

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

def create_training_data(): 
    training_data = []   
    category =  CATEGORIES[0]
    path = os.path.join('data/', category)

    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path,img))
            new_img_array = cv2.resize(img_array, (img_size, img_size))
            training_data.append([new_img_array])
        except Exception:
            print('Building training data for ' + str(category))
            
    random.shuffle(training_data)
    X = np.array(training_data).reshape(-1, img_size, img_size, 3)        
    X = X/255.0
    return X

def build_generator(seed_size, channels):
    model = Sequential([
        Dense(4*4*256,activation="relu",input_dim=seed_size),
        Reshape((4,4,256)),
        UpSampling2D(),
        Conv2D(256,kernel_size=3,padding="same"),
        BatchNormalization(momentum=0.8),
        Activation("relu"),
        UpSampling2D(),
        Conv2D(256,kernel_size=3,padding="same"),
        BatchNormalization(momentum=0.8),
        Activation("relu")
    ])
   
    # Output resolution, additional upsampling
    for _ in range(GENERATE_RES):
      model.add(UpSampling2D())
      model.add(Conv2D(128,kernel_size=3,padding="same"))
      model.add(BatchNormalization(momentum=0.8))
      model.add(Activation("relu"))

    # Final CNN layer
    model.add(Conv2D(channels,kernel_size=3,padding="same"))
    model.add(Activation("tanh"))

    input = Input(shape=(seed_size,))
    generated_image = model(input)

    return Model(input, generated_image)

def build_discriminator(image_shape):
    model = Sequential([
        Conv2D(32, kernel_size=3, strides=2, input_shape=image_shape, padding="same"),
        LeakyReLU(alpha=0.2),
        Dropout(0.25),
        Conv2D(64, kernel_size=3, strides=2, padding="same"),
        ZeroPadding2D(padding=((0,1),(0,1))),
        BatchNormalization(momentum=0.8),
        LeakyReLU(alpha=0.2),
        Dropout(0.25),
        Conv2D(128, kernel_size=3, strides=2, padding="same"),
        BatchNormalization(momentum=0.8),
        LeakyReLU(alpha=0.2),
        Dropout(0.25),
        Conv2D(256, kernel_size=3, strides=1, padding="same"),
        BatchNormalization(momentum=0.8),
        LeakyReLU(alpha=0.2),
        Dropout(0.25),
        Conv2D(512, kernel_size=3, strides=1, padding="same"),
        BatchNormalization(momentum=0.8),
        LeakyReLU(alpha=0.2),
        Dropout(0.25),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])

    input_image = Input(shape=image_shape)
    validity = model(input_image)
    return Model(input_image, validity)

def save_images(cnt,noise):
      image_array = np.full(( 
      PREVIEW_MARGIN + (PREVIEW_ROWS * (GENERATE_SQUARE+PREVIEW_MARGIN)), 
      PREVIEW_MARGIN + (PREVIEW_COLS * (GENERATE_SQUARE+PREVIEW_MARGIN)), 3), 
      255, dtype=np.uint8)
  
  generated_images = generator.predict(noise)

  generated_images = 0.5 * generated_images + 0.5

  image_count = 0
  for row in range(PREVIEW_ROWS):
      for col in range(PREVIEW_COLS):
        r = row * (GENERATE_SQUARE+16) + PREVIEW_MARGIN
        c = col * (GENERATE_SQUARE+16) + PREVIEW_MARGIN
        image_array[r:r+GENERATE_SQUARE,c:c+GENERATE_SQUARE] = generated_images[image_count] * 255
        image_count += 1

          
  output_path = os.path.join(DATA_PATH,'output')
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  
  filename = os.path.join(output_path,f"train-{cnt}.png")
  im = Image.fromarray(image_array)
  im.save(filename)

  image_shape = (GENERATE_SQUARE,GENERATE_SQUARE,IMAGE_CHANNELS)


training_data = create_training_data()

optimizer = Adam(1.5e-4,0.5) # learning rate and momentum adjusted from paper

discriminator = build_discriminator(image_shape)
discriminator.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=["accuracy"])
generator = build_generator(SEED_SIZE,IMAGE_CHANNELS)

random_input = Input(shape=(SEED_SIZE,))

generated_image = generator(random_input)

discriminator.trainable = False

validity = discriminator(generated_image)

combined = Model(random_input,validity)
combined.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=["accuracy"])

y_real = np.ones((BATCH_SIZE,1))
y_fake = np.zeros((BATCH_SIZE,1))

fixed_seed = np.random.normal(0, 1, (PREVIEW_ROWS * PREVIEW_COLS, SEED_SIZE))

cnt = 1
for epoch in range(EPOCHS):
    idx = np.random.randint(0,training_data.shape[0],BATCH_SIZE)
    x_real = training_data[idx]

    # Generate some images
    seed = np.random.normal(0,1,(BATCH_SIZE,SEED_SIZE))
    x_fake = generator.predict(seed)

    # Train discriminator on real and fake
    discriminator_metric_real = discriminator.train_on_batch(x_real,y_real)
    discriminator_metric_generated = discriminator.train_on_batch(x_fake,y_fake)
    discriminator_metric = 0.5 * np.add(discriminator_metric_real,discriminator_metric_generated)
    
    # Train generator on Calculate losses
    generator_metric = combined.train_on_batch(seed,y_real)
    
    # Time for an update?
    if epoch % SAVE_FREQ == 0:
        save_images(cnt, fixed_seed)
        cnt += 1
        print(f"Epoch {epoch}, Discriminator accuarcy: {discriminator_metric[1]}, Generator accuracy: {generator_metric[1]}")
        
generator.save(os.path.join(DATA_PATH,"face_generator.h5"))

