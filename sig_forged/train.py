
import numpy as np
##from numpy import asarray
import os 
import glob
##from PIL import Image
##import secrets
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Lambda
# from keras.layers import Dense, Conv2D, Input, MaxPool2D, Flatten, merge
import tensorflow.keras.backend as K
import tensorflow as tf
import cv2
import random
import secrets 

##base_real = '/content/main2/real'
##base_forged = '/content/main2/forged'


base = 'D:/Projects/signature_forged_detection/main'

classes = os.listdir(base)

##images = glob.glob(base, '*.png')

images = glob.glob('D:/Projects/signature_forged_detection/main/**/*.png',  
                   recursive = True) 

print('images')
print(len(images))

##real = os.listdir(base_real)
##forged = os.listdir(base_forged)

real_images = []
forged_images = []
image_shape = (800,800, 1)
labels = []
img_shape = 800
real_list = np.empty(shape=(2,5), dtype= object) 
forged_list = np.empty(shape=(2,5), dtype=object)
# pairs = []
img_shape = (28, 28,1)
 
##targets = []
     


def build_siamese_model(inputshape):

    inputs = Input(inputshape)
    x = Conv2D(64, (2, 2), padding="same", activation="relu")(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)
	# second set of CONV => RELU => POOL => DROPOUT layers
    x = Conv2D(64, (2, 2), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    pooledOutput = GlobalAveragePooling2D()(x)
    outputs = Dense(48)(pooledOutput)
	# build the model
    model = Model(inputs, outputs)
    # return the model to the calling function
    print('haha')
    return model


def euclidean_distance(vectors):
	# unpack the vectors into separate lists
	(featsA, featsB) = vectors
	# compute the sum of squared distances between the vectors
	sumSquared = K.sum(K.square(featsA - featsB), axis=1,
		keepdims=True)
	# return the euclidean distance between the vectors
	return K.sqrt(K.maximum(sumSquared, K.epsilon()))


def siamese_datagen():
    pairs = [np.zeros(( 2*len(images), 800, 800, 1)) for i in range(2)]
    targets = np.array([1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0])
    for i in range(2*len(images)):
      print(2*len(images))
      for j in range(len(classes)):
        
        real_forged = os.path.join(base, classes[j])
        real_images = os.path.join(real_forged, 'real')
        forged_images = os.path.join(real_forged, 'forged') 

        for real in glob.glob(os.path.join(real_images, '*.png')):

          img1 = cv2.imread(real)
          gray_image1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
          resized_img1= cv2.resize(gray_image1, (800, 800))

          resized_img1 = resized_img1.reshape(800,800,1)
          print(resized_img1.shape)
          print('hotay')
          print(i)
          pairs[0][i,:,:,:]  = resized_img1

          random_img = glob.glob(os.path.join(real_images, '*.png'))
          img2 = cv2.imread(secrets.choice(random_img))
          gray_image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
          resized_img2= cv2.resize(gray_image2, (800, 800))

          resized_img2 = resized_img2.reshape(800,800,1)
          pairs[1][i,:,:,:]  = resized_img2
          ##targets.append('1')

          random_forged_img = glob.glob(os.path.join(forged_images, '*.png'))
          img2 = cv2.imread(secrets.choice(random_forged_img))
          gray_image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
          resized_img2= cv2.resize(gray_image2, (800, 800))
          
          resized_img2 = resized_img2.reshape(800,800,1)

          pairs[0][i,:,:,:] = resized_img1
          pairs[1][i,:,:,:] = resized_img2
          
          ##targets.append('0')
          print('targets')
          print(targets)




    
    
    
    
  #  ' # pairs = np.empty((2, 800, 800), dtype=np.float32)
  #   # pairs = pairs.reshape(2, 800, 800, 1)    
  #   for i in range(batch_size):
  #       print('belayachi naagin')
  #       for k in range(len(real)):
  #           print('k')
  #           print(k)

  #           img1 = cv2.imread(random.choice(real_images[k]))
  #           gray_image1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
  #           resized_img1= cv2.resize(gray_image1, (800, 800))
                
  #           resized_img1 = resized_img1.reshape(800,800,1)
  #           print(resized_img1.shape)
  #           pairs[0][i,:,:,:]  = resized_img1

                
  #           print(i)
  #           if i <= batch_size:
  #               img2 = cv2.imread(random.choice(real_images[k]))
  #               gray_image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
  #               resized_img2= cv2.resize(gray_image2, (800, 800))
  #               resized_img2 = resized_img2.reshape(800,800,1)
                        
  #               pairs[1][i, :, :, :]  = resized_img2
  #               print('hahaha')
  #               np.append(targets, '1')
  #           else:
                        
  #               img2 = cv2.imread(random.choice(forged_images[k]))
  #               print(img2)
                    
  #               gray_image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
  #               resized_img2= cv2.resize(gray_image2, (800, 800))
  #               resized_img2 = resized_img2.reshape(800,800,1)

  #               pairs[1][i, :, :, :]  = resized_img2 
  #               np.append(targets, '0')'
    print('targets')
    print(targets)                        
    return pairs, np.asarray(targets)


                 

real_array = []
print('haha')
                  


# training_data = siamese_datagen(real_images, forged_images, batch_size= 2)    


(inputs, targets) = siamese_datagen()


featureExtractor = build_siamese_model(image_shape)
imgA = Input(shape = image_shape)
imgB = Input(shape = image_shape)
featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)

distance = Lambda(euclidean_distance)([featsA, featsB])
outputs = Dense(1, activation="sigmoid")(distance)
model = Model(inputs=[imgA, imgB], outputs=outputs)

model.compile(loss="binary_crossentropy", optimizer="adam",
	metrics=["accuracy"])
# train the model
print("[INFO] training model...")
history = model.fit(
 	inputs, targets[:],
 	batch_size=16, 
 	epochs=100)

model.save('D:/Projects/signature_forged_detection/siamese_model')


        
