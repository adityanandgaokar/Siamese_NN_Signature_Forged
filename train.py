# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 14:39:55 2021

@author: Aditya
"""


import numpy as np
##from numpy import asarray
import os 
import glob
##import secrets
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Flatten
# from keras.layers import Dense, Conv2D, Input, MaxPool2D, Flatten, merge
import tensorflow.keras.backend as K
import cv2
from keras.regularizers import l2
from keras.optimizers import Adam, RMSprop
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate, Dropout
from keras.models import Model

from keras.layers.normalization import BatchNormalization

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


base = '/content/drive/MyDrive/main'

classes = os.listdir(base)

##images = glob.glob(base, '*.png')

images = glob.glob('/content/drive/MyDrive/main/**/*.png',  
                   recursive = True) 

print('images')
print(len(images))

##real = os.listdir(base_real)
##forged = os.listdir(base_forged)

real_images = []
forged_images = []
img_h, img_w = 150, 200
labels = []
img_shape = 28
real_list = np.empty(shape=(2,5), dtype= object) 
forged_list = np.empty(shape=(2,5), dtype=object)
# pairs = []
img_shape = (28, 28,1)
 
##targets = []
     


def build_siamese_model(input_shape):
  
  model = Sequential()
  
  model.add(Conv2D(96, kernel_size=(11, 11), activation='relu', name='conv1_1', strides=4, input_shape= input_shape, 
                        kernel_initializer='glorot_uniform'))
  model.add(BatchNormalization(epsilon=1e-06, axis=1, momentum=0.9))
  model.add(MaxPooling2D((3,3), strides=(2, 2)))    
  model.add(ZeroPadding2D((2, 2)))
    
  model.add(Conv2D(256, kernel_size=(5, 5), activation='relu', name='conv2_1', strides=1, kernel_initializer='glorot_uniform'))
  model.add(BatchNormalization(epsilon=1e-06, axis=1, momentum=0.9))
  model.add(MaxPooling2D((3,3), strides=(2, 2)))
  model.add(Dropout(0.3))# added extra
  model.add(ZeroPadding2D((1, 1)))
    
  model.add(Conv2D(384, kernel_size=(3, 3), activation='relu', name='conv3_1', strides=1, kernel_initializer='glorot_uniform'))
  model.add(ZeroPadding2D((1, 1)))
    
  model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', name='conv3_2', strides=1, kernel_initializer='glorot_uniform'))    
  model.add(MaxPooling2D((3,3), strides=(2, 2)))
  model.add(Dropout(0.3))# added extra
  model.add(Flatten(name='flatten'))
  model.add(Dense(1024, kernel_regularizer=l2(0.0005), activation='relu', kernel_initializer='glorot_uniform'))
  model.add(Dropout(0.5))
    
  model.add(Dense(256, kernel_regularizer=l2(0.0005), activation='relu', kernel_initializer='glorot_uniform')) # softmax changed to relu
    
  return model

def euclidean_distance(vectors):
	# unpack the vectors into separate lists
	featsA, featsB = vectors
	# compute the sum of squared distances between the vectors
	return K.sqrt(K.sum(K.square(featsA - featsB), axis=1, keepdims=True))


def euclidean_distance_output_shape(shapes):
  shape1, shape2 = shapes
  return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):

  margin = 1

  loss = K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

  return loss





def image_read(path):
  image = cv2.imread(path, 0)
  image = cv2.resize(image, (img_w, img_h))
  image = np.array(image, dtype= np.float64)
  image /= 255
  image = image.reshape(img_h, img_w, 1)

  return image


training_dir =  '/content/drive/MyDrive/sign_data/train'
training_csv = '/content/drive/MyDrive/sign_data/train_data.csv'
test_csv = '/content/drive/MyDrive/sign_data/test_data.csv'
test_dir = '/content/drive/MyDrive/sign_data/test'

def siamese_datagen(training_dir, training_csv, batch_size = 16):
    while True:
      targets = np.zeros((batch_size,))
      training_df = pd.read_csv(training_csv)
      pairs = [np.zeros((batch_size, img_h, img_w, 1)) for i in range(2)]
      p = 0
      for i in range(0, len(training_df)):
        image1_path = os.path.join(training_dir, training_df.iat[i, 0])
        image2_path = os.path.join(training_dir, training_df.iat[i, 1])

        img1 = image_read(image1_path)
        img2 = image_read(image2_path)

        pairs[0][p,:,:,:] = img1
        pairs[1][p,:,:,:] = img2

        targets[p] = training_df.iat[i, 2]

        p += 1

        if p == batch_size:
          yield pairs, targets
          p = 0
          pairs = [np.zeros((batch_size, img_h, img_w, 1)) for i in range(2)]
          targets = np.zeros((batch_size,))
 
  
                        


                 

# training_data = siamese_datagen(real_images, forged_images, batch_size= 2)    

image_shape = (img_h, img_w, 1)

featureExtractor = build_siamese_model(image_shape)
imgA = Input(shape = (image_shape))
imgB = Input(shape = (image_shape))
featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)

distance = Lambda(euclidean_distance, output_shape = euclidean_distance_output_shape)([featsA, featsB])

model = Model(inputs=[imgA, imgB], outputs=distance)  

batch_sz = 256
training_samples = 23206
validation_samples = 5748

rms = RMSprop(lr=0.01, rho=0.9, epsilon=1e-08)
model.compile(loss=contrastive_loss, optimizer=rms)


callbacks = [
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.000001, verbose=1),
    ModelCheckpoint('/content/siamese_network.h5', verbose=1, save_weights_only=True)
]



# train the model
print("[INFO] training model...")
history = model.fit_generator(siamese_datagen(training_dir, training_csv, batch_sz),
 	 validation_data= siamese_datagen(test_dir, test_csv, batch_sz),
   epochs=50,
   steps_per_epoch = training_samples//batch_sz,
   validation_steps = validation_samples // batch_sz,
  callbacks = callbacks)

