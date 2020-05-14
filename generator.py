############################## Load package ##############################

import os
import cv2
import sys
import glob
import math
import json
import time
import random
import shutil
import argparse
import requests 
import functools
import numpy as np
from numpy import asarray
from numpy import moveaxis
from numpy import expand_dims
from PIL import Image
from imutils import paths
from scipy.linalg import sqrtm
import tensorflow as tf
from tensorflow.python.ops import array_ops
import keras
from keras import initializers
from keras import backend as K
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.preprocessing import image
from keras.models import Sequential, Model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, Conv2DTranspose, Reshape, Flatten
from keras.layers import Input, Dense, LeakyReLU, BatchNormalization, Dropout
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.inception_v3 import InceptionV3, preprocess_input


############################## Define parse ##############################

parser = argparse.ArgumentParser()
parser.add_argument("-base_dir", "--base_dir", type=str, default=None, help='Image path')
parser.add_argument("-training", "--training", type=str, default='False', help='Selection for Generate image')
parser.add_argument("-rotation_range", "--rotation_range", type=int, default=90, help='Rotation range 0-90')
parser.add_argument("-shear_range", "--shear_range", type=float, default=0.1, help='Shear range 0.1-0.2')
parser.add_argument("-horizontal_flip", "--horizontal_flip", type=str, default='True', help='Select True or False for Horizontal flip')
parser.add_argument("-vertical_flip", "--vertical_flip", type=str, default='True', help='Select True or False for Vertical flip')
parser.add_argument("-jobid", "--jobid", type=str, default=None, help='Write anything')
parser.add_argument("-url_prefix", "--url_prefix", type=str, default=None, help='URL address')
args = parser.parse_args()


#################### Define funcion for image Generation ####################

multiply_number_for_2_increaing = 400
height = 48
width = 48
channels = 1

def resize_down_size_image(base_directory, original_images_dir):
#    original_image_path = os.path.join(base_directory, 'original_images')
    original_image_path = original_images_dir
    list_image_paths = list(paths.list_images(original_image_path))
    save_path = os.path.join(base_directory, '1_1_down_size_images')
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    for inx, path in enumerate(list_image_paths):
        image = Image.open(path)
        resize_img = image.resize((48, 48))
        image_label = path.split(os.path.sep)[-1]
        resize_img.save(save_path + '/' + image_label)

def increasing_down_size_image(base_directory):
    image_path_for_increasing = os.path.join(base_directory, '1_1_down_size_images')
    list_image_paths_for_increasing = list(paths.list_images(image_path_for_increasing))
    save_path = os.path.join(base_directory, '1_2_increasing_down_size_images')
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
        
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=90,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    
    image_array = []
    for index, img in enumerate(list_image_paths_for_increasing):
        image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        image = img_to_array(image)
        image_array.append(image)
    image_array = np.array(image_array, dtype="float") / 128. - 1
    
    train_datagen.fit(image_array)
    
    multiply_number = math.ceil(multiply_number_for_2_increaing / len(list_image_paths_for_increasing))
    
    i = 0
    for batch in train_datagen.flow(image_array,
                                    batch_size = len(list_image_paths_for_increasing),
                                    save_to_dir = save_path,
                                    save_prefix='bw',
                                    save_format='png'):
        i += 1
        if i > (multiply_number - 1):
            break

            
def generate_image(base_directory):
    image_path_for_generating = os.path.join(base_directory, '1_2_increasing_down_size_images')
    list_image_paths_for_generating = list(paths.list_images(image_path_for_generating))
    save_path = os.path.join(base_directory, '1_3_generated_images')
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
        
    random.shuffle(list_image_paths_for_generating)
    
    train_datas = []
    
    for index, img_path in enumerate(list_image_paths_for_generating):
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = img_to_array(image)
        train_datas.append(image)
    
    train_datas = np.array(train_datas)
    
    x_train = train_datas.reshape((train_datas.shape[0],) + (height, width, channels)).astype('float32')
    X_train = (x_train - 127.5) / 127.5
    
    ## define model
    
    # latent space dimension
    latent_dim = 100
    # Image demension
    init = initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None)
    
    # Generator network
    generator = Sequential()
    # FC:
    generator.add(Dense(144, input_shape=(latent_dim,), kernel_initializer=init))
    # FC:
    generator.add(Dense(12*12*128))
    generator.add(Reshape((12, 12, 128)))
    generator.add(Dropout(0.5))
    # Conv 1:
    generator.add(Conv2DTranspose(128, kernel_size=2, strides=2, padding='same'))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(LeakyReLU(0.2))
    # Conv 2:
    generator.add(Conv2DTranspose(128, kernel_size=2, strides=2, padding='same'))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(LeakyReLU(0.2))
    # Conv 3:
    generator.add(Conv2DTranspose(64, kernel_size=2, strides=1, padding='same'))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(LeakyReLU(0.2))
    # Conv 4:
    generator.add(Conv2DTranspose(1, kernel_size=2, strides=1, padding='same', activation='tanh'))
    
    # Discriminator network
    discriminator = Sequential()
    # Conv 1:
    discriminator.add(Conv2D(64, kernel_size=1, strides=1, padding='same', input_shape=(48, 48, 1), kernel_initializer=init))
    discriminator.add(LeakyReLU(0.2))
    # Conv 2:
    discriminator.add(Conv2D(64, kernel_size=2, strides=1, padding='same'))
    discriminator.add(BatchNormalization(momentum=0.8))
    discriminator.add(LeakyReLU(0.2))
    # Conv 3:
    discriminator.add(Conv2D(128, kernel_size=2, strides=2, padding='same'))
    discriminator.add(BatchNormalization(momentum=0.8))
    discriminator.add(LeakyReLU(0.2))
    # Conv 4:
    discriminator.add(Conv2D(256, kernel_size=2, strides=2, padding='same'))
    discriminator.add(BatchNormalization(momentum=0.8))
    discriminator.add(LeakyReLU(0.2))
    # FC
    discriminator.add(Flatten())
    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dense(512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.5))
    # Output
    discriminator.add(Dense(1, activation='sigmoid'))    
    
    # Optimizer
    optimizer = Adam(lr=0.0002, beta_1=0.5)
    discriminator.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
    
    discriminator.trainable = False
    z = Input(shape=(latent_dim,))
    img = generator(z)
    decision = discriminator(img)
    d_g = Model(inputs=z, outputs=decision)
    d_g.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
    
    epochs = 150
    batch_size = 32
    smooth = 0.1
    
    real = np.ones(shape=(batch_size, 1))
    fake = np.zeros(shape=(batch_size, 1))
    
    d_loss = []
    d_g_loss = []
    
    for e in range(epochs):
        print(e, file=sys.stderr, flush=True)
        for i in range(len(X_train) // batch_size):
            # Train Discriminator weights
            discriminator.trainable = True
            # Real samples
            X_batch = X_train[i*batch_size:(i+1)*batch_size]
            d_loss_real = discriminator.train_on_batch(x=X_batch, y=real*(1-smooth))
            
            # Fake samples
            z = np.random.normal(loc=0, scale=1, size=(batch_size, latent_dim))
            X_fake = generator.predict_on_batch(z)
            d_loss_fake = discriminator.train_on_batch(x=X_fake, y=fake)
            
            # Discriminator loss
            d_loss_batch = 0.5 * (d_loss_real[0] + d_loss_fake[0])
            
            # Train Generator weights
            discriminator.trainable = False
            d_g_loss_batch = d_g.train_on_batch(x=z, y=real)
            
            samples = batch_size
            
            if e == (epochs-1):
                for k in range(len(X_batch)):
                    x_fake = generator.predict(np.random.normal(loc=0, scale=1, size=(samples, latent_dim)))
                    
                    img = keras.preprocessing.image.array_to_img(x_fake[k] * 255., scale=False)
                    img.save(os.path.join(save_path, 'generated_wafer' + str(e) + '_' + str(i) + '_' + str(k) +'.png'))
        d_loss.append(d_loss_batch)
        d_g_loss.append(d_g_loss_batch[0])

        
def save_binarized_image(base_directory):
    image_path_for_binarization = os.path.join(base_directory, '1_3_generated_images')
    list_image_path_for_binarization = list(paths.list_images(image_path_for_binarization))
    save_path = os.path.join(base_directory, '1_4_binarized_images')
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    for index, path in enumerate(list_image_path_for_binarization):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        ret, threshold_image = cv2.threshold(image, 140, 255, cv2.THRESH_BINARY)
        img = Image.fromarray(threshold_image)
        img.save(os.path.join(save_path, 'binarization_wafer' + '_' + str(index) + '(140)' + '.png'))

def resize_upsize_image(base_directory, image_name):
    image_path_for_upsize = os.path.join(base_directory, '1_4_binarized_images')
    list_image_path_for_upsize = list(paths.list_images(image_path_for_upsize))
#    save_path = os.path.join(base_directory, '1_5_up_size_images')
    save_path = os.path.join(base_directory, 'AI_Generated_Images_' + image_name)
    
#    if os.path.isdir(save_path):
#        shutil.rmtree(save_path)
#    if not os.path.isdir(save_path):
#        os.makedirs(save_path)
        
    if os.path.isdir(save_path):
        image_name_number = 2
        save_path = os.path.join(base_directory, 'AI_Generated_images_' + image_name + '(' + str(image_name_number) + ')')
        if os.path.isdir(save_path):
            image_name_number = 3
            save_path = os.path.join(base_directory, 'AI_Generated_images_' + image_name + '(' + str(image_name_number) + ')')
            if os.path.isdir(save_path):
                image_name_number = 4
                save_path = os.path.join(base_directory, 'AI_Generated_images_' + image_name + '(' + str(image_name_number) + ')')
                if os.path.isdir(save_path):
                    image_name_number = 5
                    save_path = os.path.join(base_directory, 'AI_Generated_images_' + image_name + '(' + str(image_name_number) + ')')
                    if os.path.isdir(save_path):
                        image_name_number = 6
                        save_path = os.path.join(base_directory, 'AI_Generated_images_' + image_name + '(' + str(image_name_number) + ')')
                        if os.path.isdir(save_path):
                            image_name_number = 7
                            save_path = os.path.join(base_directory, 'AI_Generated_images_' + image_name + '(' + str(image_name_number) + ')')
                            if os.path.isdir(save_path):
                                image_name_number = 8
                                save_path = os.path.join(base_directory, 'AI_Generated_images_' + image_name + '(' + str(image_name_number) + ')')
                                if os.path.isdir(save_path):
                                    image_name_number = 9
                                    save_path = os.path.join(base_directory, 'AI_Generated_images_' + image_name + '(' + str(image_name_number) + ')')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
        
    for inx, path in enumerate(list_image_path_for_upsize):
        image = Image.open(path)
        resize_img = image.resize((120, 120))
        image_label = path.split(os.path.sep)[-1]
        resize_img.save(save_path + '/' + image_label)
    return save_path

########## Incressing image for compare ##########
def increasing_image_for_compare(base_directory, original_images_dir):
#    image_path_for_increasing = os.path.join(base_directory, 'original_images')
    image_path_for_increasing = original_images_dir
    list_image_paths_for_increasing = list(paths.list_images(image_path_for_increasing))
    save_path = os.path.join(base_directory, '3_1_original_size_images_for_compare')
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
        
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=90,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    
    image_array = []
    for index, img in enumerate(list_image_paths_for_increasing):
        image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        image = img_to_array(image)
        image_array.append(image)
    image_array = np.array(image_array, dtype="float") / 128. - 1
    
    train_datagen.fit(image_array)
    
    multiply_number = math.ceil(multiply_number_for_2_increaing / len(list_image_paths_for_increasing))
    
    i = 0
    for batch in train_datagen.flow(image_array,
                                    batch_size = len(list_image_paths_for_increasing),
                                    save_to_dir = save_path,
                                    save_prefix='bw',
                                    save_format='png'):
        i += 1
        if i > (multiply_number - 1):
            break


############################## if training == "False" ##############################
def training_false_increasing_image(base_directory, original_images_dir, rotation_range, shear_range, horizontal_flip, vertical_flip, image_name):
#    image_path_for_increasing = os.path.join(base_directory, 'original_images')
    image_path_for_increasing = original_images_dir
    list_image_paths_for_increasing = list(paths.list_images(image_path_for_increasing))
#    save_path = os.path.join(base_directory, '2_1_increasing_original_size_images')
    save_path = os.path.join(base_directory, 'User_Generated_Images_' + image_name)

#    if os.path.isdir(save_path):
#        shutil.rmtree(save_path)
#        save_path = os.path.join(base_directory, 'User_Generated_images_' + image_name + ' (' + str(image_name_number) + ')')
#    if not os.path.isdir(save_path):
#        os.makedirs(save_path)
    if os.path.isdir(save_path):
        image_name_number = 2
        save_path = os.path.join(base_directory, 'User_Generated_images_' + image_name + '(' + str(image_name_number) + ')')
        if os.path.isdir(save_path):
            image_name_number = 3
            save_path = os.path.join(base_directory, 'User_Generated_images_' + image_name + '(' + str(image_name_number) + ')')
            if os.path.isdir(save_path):
                image_name_number = 4
                save_path = os.path.join(base_directory, 'User_Generated_images_' + image_name + '(' + str(image_name_number) + ')')
                if os.path.isdir(save_path):
                    image_name_number = 5
                    save_path = os.path.join(base_directory, 'User_Generated_images_' + image_name + '(' + str(image_name_number) + ')')
                    if os.path.isdir(save_path):
                        image_name_number = 6
                        save_path = os.path.join(base_directory, 'User_Generated_images_' + image_name + '(' + str(image_name_number) + ')')
                        if os.path.isdir(save_path):
                            image_name_number = 7
                            save_path = os.path.join(base_directory, 'User_Generated_images_' + image_name + '(' + str(image_name_number) + ')')
                            if os.path.isdir(save_path):
                                image_name_number = 8
                                save_path = os.path.join(base_directory, 'User_Generated_images_' + image_name + '(' + str(image_name_number) + ')')
                                if os.path.isdir(save_path):
                                    image_name_number = 9
                                    save_path = os.path.join(base_directory, 'User_Generated_images_' + image_name + '(' + str(image_name_number) + ')')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
        
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=rotation_range,
        shear_range=shear_range,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        fill_mode='nearest'
    )
    
    image_array = []
    for index, img in enumerate(list_image_paths_for_increasing):
        image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        image = img_to_array(image)
        image_array.append(image)
    image_array = np.array(image_array, dtype="float") / 128. - 1
    
    train_datagen.fit(image_array)
    
    multiply_number = math.ceil(multiply_number_for_2_increaing / len(list_image_paths_for_increasing))
    
    i = 0
    for batch in train_datagen.flow(image_array,
                                    batch_size = len(list_image_paths_for_increasing),
                                    save_to_dir = save_path,
                                    save_prefix='bw',
                                    save_format='png'):
        i += 1
        if i > (multiply_number - 1):
            break
    return save_path


############################## function related to IS ##############################
def image_array_for_is(path_for_is):
    image_paths = list(paths.list_images(path_for_is))

    image_array = []

    for ix, path in enumerate(image_paths):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = img_to_array(image)
        image = moveaxis(image, 2, 0)
        image_array.append(image)
    image_array = np.array(image_array)
    return image_array

tfgan = tf.contrib.gan

#session=tf.compat.v1.InteractiveSession()
session=tf.InteractiveSession()

# A smaller BATCH_SIZE reduces GPU memory usage, but at the cost of a slight slowdown
BATCH_SIZE = 64
INCEPTION_URL = 'http://download.tensorflow.org/models/frozen_inception_v1_2015_12_05.tar.gz'
INCEPTION_FROZEN_GRAPH = 'inceptionv1_for_inception_score.pb'

# Run images through Inception.
#inception_images = tf.compat.v1.placeholder(tf.float32, [None, 3, None, None])
inception_images = tf.placeholder(tf.float32, [None, 3, None, None])
def inception_logits(images = inception_images, num_splits = 1):
    images = tf.transpose(images, [0, 2, 3, 1])
    size = 299
#    images = tf.compat.v1.image.resize_bilinear(images, [size, size])
    images = tf.image.resize_bilinear(images, [size, size])
    generated_images_list = array_ops.split(images, num_or_size_splits = num_splits)
    logits = tf.map_fn(
        fn = functools.partial(
             tfgan.eval.run_inception, 
             default_graph_def_fn = functools.partial(
             tfgan.eval.get_graph_def_from_url_tarball, 
             INCEPTION_URL, 
             INCEPTION_FROZEN_GRAPH, 
             os.path.basename(INCEPTION_URL)), 
             output_tensor = 'logits:0'),
        elems = array_ops.stack(generated_images_list),
        parallel_iterations = 1,
        back_prop = False,
        swap_memory = True,
        name = 'RunClassifier')
    logits = array_ops.concat(array_ops.unstack(logits), 0)
    return logits

logits=inception_logits()

def get_inception_probs(inps):
    n_batches = int(np.ceil(float(inps.shape[0]) / BATCH_SIZE))
    preds = np.zeros([inps.shape[0], 1000], dtype = np.float32)
    for i in range(n_batches):
        inp = inps[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] / 255. * 2 - 1
        preds[i * BATCH_SIZE : i * BATCH_SIZE + min(BATCH_SIZE, inp.shape[0])] = session.run(logits,{inception_images: inp})[:, :1000]
    preds = np.exp(preds) / np.sum(np.exp(preds), 1, keepdims=True)
    return preds

def preds2score(preds, splits=10):
    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        q = np.expand_dims(np.mean(part, 0), 0)
        kl = part * (np.log(part / q)) + (1 - part) * np.log((1 - part) / (1 - q))
        kl = np.mean(kl)
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)

def preds2score(preds, splits=10):
    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)

def get_inception_score(images, splits=10):
    assert(type(images) == np.ndarray)
    assert(len(images.shape) == 4)
    assert(images.shape[1] == 3)
    assert(np.min(images[0]) >= 0 and np.max(images[0]) > 10), 'Image values should be in the range [0, 255]'
    print('Calculating Inception Score with %i images in %i splits' % (images.shape[0], splits), file=sys.stderr, flush=True)
    start_time=time.time()
    preds = get_inception_probs(images)
    mean, std = preds2score(preds, splits)
    print('Inception Score calculation time: %f s' % (time.time() - start_time), file=sys.stderr, flush=True)
    return mean, std


############################## function related to FID ##############################
def resize_wafer(x):
    x_list = []
    for i in range(x.shape[0]):
        if training == "True":
            img = image.array_to_img(x[i, :, :, :].reshape(48, 48, -1))
        else:
            img = image.array_to_img(x[i, :, :, :].reshape(120, 120, -1))
        img = img.resize(size=(299, 299), resample=Image.LANCZOS)
        x_list.append(image.img_to_array(img))
    return np.array(x_list)

def cal_h(x, resizer, batch_size=64):
    model = InceptionV3()
    model4fid = Model(inputs=model.input, outputs=model.get_layer("avg_pool").output)
    
    r = None
    n_batch = (x.shape[0]+batch_size-1) // batch_size
    for j in range(n_batch):
        x_batch = resizer(x[j*batch_size:(j+1)*batch_size, :, :, :])
        r_batch = model4fid.predict(preprocess_input(x_batch))
        r = r_batch if r is None else np.concatenate([r, r_batch], axis=0)
    return r

def wafer_h(n_train, n_val, list_real_wafer, list_generated_wafer):
    
    real_data = []
    fake_data = []
    
    x = [0, 0]
    h = [0, 0]
    n = [n_train, n_val]
    
    for ix, image_path in enumerate(list_real_wafer):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = img_to_array(image)
        real_data.append(image)
    real_data = np.array(real_data, dtype="float") / 255.0
    x[0] = np.tile(real_data, (1, 1, 1, 3))
    
    for ix, image_path in enumerate(list_generated_wafer):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = img_to_array(image)
        fake_data.append(image)
    fake_data = np.array(fake_data, dtype="float") / 255.0
    x[1] = np.tile(fake_data, (1, 1, 1, 3))
    
    for i in range(2):
        h[i] = cal_h(x[i][0:n[i], :, :, :], resize_wafer)
    return h[0], h[1]

def mean_cov(x):
    mean = np.mean(x, axis=0)
    sigma = np.cov(x, rowvar=False)
    return mean, sigma

def frechet_distance(m1, c1, m2, c2):
    return np.sum((m1 - m2)**2) + np.trace(c1 + c2 -2*(sqrtm(np.dot(c1, c2))))

def calculate_fid_score(h1, h2):
#    model = InceptionV3()
#    model4fid = Model(inputs=model.input, outputs=model.get_layer("avg_pool").output)
    
    m1, c1 = mean_cov(h1)
    m2, c2 = mean_cov(h2)
    return frechet_distance(m1, c1, m2, c2)
  
    
############################## Execute ##############################

if __name__ == "__main__":
    original_images_dir = args.base_dir    
#    __file__ = base_dir_for_split
    base_dir = os.path.realpath(original_images_dir).rsplit('/', 1)[0] + '/'
    print(original_images_dir, file=sys.stderr, flush=True)
    print(base_dir, file=sys.stderr, flush=True)
    
    image_name = os.path.basename(os.path.normpath(original_images_dir))
    print(image_name, file=sys.stderr, flush=True)
    
    training = args.training
    
    jobid = args.jobid
    
    URL = args.url_prefix
    URL = URL + '/generator/updateStatus'
    headers={'Content-type':'application/json', 'Accept':'application/json'}

    if training == "True":
        ### False increaing image
        data = {'step': 'step-1', 'status': 'RUNNING', 'jobid': jobid, 'training': 'True'}
        res = requests.post(URL, json=data, headers=headers)
        print(res, file=sys.stderr, flush=True)
        resize_down_size_image(base_dir, original_images_dir)
#        with open(os.path.join(base_dir,"ai_status.txt"), "w") as f:
#            f.write("Finished 1/7 ")
        data = {'step': 'step-2', 'status': 'RUNNING', 'jobid': jobid, 'training': 'True'}
        res = requests.post(URL, json=data, headers=headers)
        print(res, file=sys.stderr, flush=True)
        increasing_down_size_image(base_dir)
#        with open(os.path.join(base_dir,"ai_status.txt"), "w") as f:
#            f.write("Finished 2/7")
        data = {'step': 'step-3', 'status': 'RUNNING', 'jobid': jobid, 'training': 'True'}
        res = requests.post(URL, json=data, headers=headers)
        print(res, file=sys.stderr, flush=True)
        generate_image(base_dir)

#        with open(os.path.join(base_dir,"ai_status.txt"), "w") as f:
#            f.write("Finished 3/7")
        data = {'step': 'step-4', 'status': 'RUNNING', 'jobid': jobid, 'training': 'True'}
        res = requests.post(URL, json=data, headers=headers)
        print(res, file=sys.stderr, flush=True)
        save_binarized_image(base_dir)
#        with open(os.path.join(base_dir,"ai_status.txt"), "w") as f:
#            f.write("Finished 4/7")
        data = {'step': 'step-5', 'status': 'RUNNING', 'jobid': jobid, 'training': 'True'}
        res = requests.post(URL, json=data, headers=headers)
        print(res, file=sys.stderr, flush=True)
        save_path = resize_upsize_image(base_dir, image_name)
#        with open(os.path.join(base_dir,"ai_status.txt"), "w") as f:
#            f.write("Finished 5/7")

        ### Increasing image for compare
#        increasing_image_for_compare(base_dir)

        ### Calculate IS
#        path_for_is = os.path.join(base_dir, '1_5_up_size_images')
#        path_for_is = os.path.join(base_dir, 'AI_Gerated_Images_' + image_name)
        path_for_is = save_path
        image_array = image_array_for_is(path_for_is)
        data = {'step': 'step-6', 'status': 'RUNNING', 'jobid': jobid, 'training': 'True'}
        res = requests.post(URL, json=data, headers=headers)
        print(res, file=sys.stderr, flush=True)
        inception_score_mean, inception_score_standard = get_inception_score(image_array, splits=10)
#        with open(os.path.join(base_dir,"ai_status.txt"), "w") as f:
#            f.write("Finished 6/7")
        print(inception_score_mean, file=sys.stderr, flush=True) 
        
        ### Calculate FID 
#        real_wafer_path = os.path.join(base_dir, '3_1_original_size_images_for_compare')
#        generated_wafer_path = os.path.join(base_dir, '1_5_up_size_images')
#        generated_wafer_path = os.path.join(base_dir, 'AI_Gerated_Images_' + image_name)
        generated_wafer_path = save_path
        real_wafer_path_for_fid = os.path.join(base_dir, '1_1_down_size_images')
        generated_wafer_path_for_fid = os.path.join(base_dir, '1_4_binarized_images')     
        
        list_real_wafer = list(paths.list_images(real_wafer_path_for_fid))
        n_train = len(list_real_wafer)
        list_generated_wafer = list(paths.list_images(generated_wafer_path_for_fid)) #416
        n_val = len(list_generated_wafer)

        h_real, h_fake = wafer_h(n_train, n_val, list_real_wafer, list_generated_wafer)
        data = {'step': 'step-7', 'status': 'RUNNING', 'jobid': jobid, 'training': 'True'}
        res = requests.post(URL, json=data, headers=headers)
        print(res, file=sys.stderr, flush=True)
        fid_score = calculate_fid_score(h_real, h_fake)
        data = {'step': 'FINISH', 'status': 'FINISH', 'jobid': jobid, 'training': 'True'}
        res = requests.post(URL, json=data, headers=headers)
        print(res, file=sys.stderr, flush=True)
        shutil.rmtree(os.path.join(base_dir, '1_1_down_size_images')) 
        shutil.rmtree(os.path.join(base_dir, '1_2_increasing_down_size_images'))  
        shutil.rmtree(os.path.join(base_dir, '1_3_generated_images'))  
        shutil.rmtree(os.path.join(base_dir, '1_4_binarized_images'))  
        fid_score = str(fid_score)[1:10]
        fid_score = float(fid_score)
#        with open(os.path.join(base_dir,"ai_status.txt"), "w") as f:
#            f.write("Finished 7/7")
        print(fid_score, file=sys.stderr, flush=True)
        score_dict = {}
        score_dict['FID'] = fid_score
        score_dict['IS'] = inception_score_mean
        score_dict = str(score_dict)
        with open(os.path.join(save_path,"result.txt"), "w") as f:
            f.write(score_dict)
    else:
        rotation_range = args.rotation_range
        shear_range = args.shear_range
        horizontal_flip = args.horizontal_flip
        vertical_flip = args.vertical_flip

        ### Auto increaing image
        data = {'step': 'step-1', 'status': 'RUNNING', 'jobid': jobid, 'training': 'False'}
        res = requests.post(URL, json=data, headers=headers)
        print(res, file=sys.stderr, flush=True)
#        training_false_increasing_image(base_dir, original_images_dir, rotation_range, shear_range, horizontal_flip, vertical_flip, image_name)
        save_path = training_false_increasing_image(base_dir, original_images_dir, rotation_range, shear_range, horizontal_flip, vertical_flip, image_name)
#        with open(os.path.join(base_dir,"user_status.txt"), "w") as f:
#            f.write("Finished 1/4")
        
        ### Increasing image for compare
        data = {'step': 'step-2', 'status': 'RUNNING', 'jobid': jobid, 'training': 'False'}
        res = requests.post(URL, json=data, headers=headers)
        print(res, file=sys.stderr, flush=True)
        increasing_image_for_compare(base_dir, original_images_dir)
#        with open(os.path.join(base_dir,"user_status.txt"), "w") as f:
#            f.write("Finished 2/4")
            
        ### Calculate IS
        data = {'step': 'step-3', 'status': 'RUNNING', 'jobid': jobid, 'training': 'False'}
        res = requests.post(URL, json=data, headers=headers)
        print(res, file=sys.stderr, flush=True)
#        path_for_is = os.path.join(base_dir, '2_1_increasing_original_size_images')
#        path_for_is = os.path.join(base_dir, 'User_Generated_Images_'+image_name)
        path_for_is = save_path
        image_array = image_array_for_is(path_for_is)
        
        inception_score_mean, inception_score_standard = get_inception_score(image_array, splits=10)
#        with open(os.path.join(base_dir,"user_status.txt"), "w") as f:
#            f.write("Finished 3/4")
        print(inception_score_mean, file=sys.stderr, flush=True) 

        ### Calculate FID
        real_wafer_path = os.path.join(base_dir, '3_1_original_size_images_for_compare')
#        generated_wafer_path = os.path.join(base_dir, '2_1_increasing_original_size_images') 
#        generated_wafer_path = os.path.join(base_dir, 'User_Generated_Images_'+image_name)
        generated_wafer_path = save_path

        list_real_wafer = list(paths.list_images(real_wafer_path))
        n_train = len(list_real_wafer) 
        list_generated_wafer = list(paths.list_images(generated_wafer_path)) 
        n_val = len(list_generated_wafer)

        h_real, h_fake = wafer_h(n_train, n_val, list_real_wafer, list_generated_wafer)
        data = {'step': 'step-4', 'status': 'RUNNING', 'jobid': jobid, 'training': 'False'}
        res = requests.post(URL, json=data, headers=headers)
        print(res, file=sys.stderr, flush=True)
        fid_score = calculate_fid_score(h_real, h_fake)
        data = {'step': 'FINISH', 'status': 'FINISH', 'jobid': jobid, 'training': 'False'}
        res = requests.post(URL, json=data, headers=headers)
        print(res, file=sys.stderr, flush=True)
        ### Delete real_wafer_path
        if os.path.isdir(real_wafer_path):
            shutil.rmtree(real_wafer_path)
        fid_score = str(fid_score)[1:10]
        fid_score = float(fid_score)
#        with open(os.path.join(base_dir,"user_status.txt"), "w") as f:
#            f.write("Finished 4/4")
        print(fid_score, file=sys.stderr, flush=True)
        score_dict = {}
        score_dict['FID'] = fid_score
        score_dict['IS'] = inception_score_mean
        score_dict = str(score_dict)
        with open(os.path.join(save_path,"result.txt"), "w") as f:
            f.write(score_dict)


#    result_output = {"generated_image_path": generated_wafer_path, "is_score": inception_score_mean, "fid_score": fid_score}
#    print(result_output, file=sys.stderr, flush=True)
#    result_json = json.dumps(result_output)
#    sys.stdout.write(result_json)
    inception_score_mean = np.float32(inception_score_mean)
    fid_score = np.float32(fid_score)

    result_output = {}
    result_output["generated_image_path"] = generated_wafer_path
    result_output["is_score"] = inception_score_mean.item()
    result_output["fid_score"] = fid_score.item()

    #print(result_output, file=sys.stdout, flush=True)
    result_json = json.dumps(result_output)
    print(result_json, file=sys.stdout, flush=True)
