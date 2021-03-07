
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
from tensorflow.keras.layers import Flatten
# from keras.layers import Dense, Conv2D, Input, MaxPool2D, Flatten, merge
import tensorflow.keras.backend as K
import tensorflow as tf
import cv2
import random
import secrets 



base = '/content/main'

classes = os.listdir(base)

##images = glob.glob(base, '*.png')

images = glob.glob('/content/main/**/*.png',  
                   recursive = True) 

print('images')
print(len(images))

##real = os.listdir(base_real)
##forged = os.listdir(base_forged)

real_images = []
forged_images = []
image_shape = (120,120, 1)
labels = []
img_shape = 28
real_list = np.empty(shape=(2,5), dtype= object) 
forged_list = np.empty(shape=(2,5), dtype=object)
# pairs = []
img_shape = (28, 28,1)
 
##targets = []
     


def build_siamese_model(inputshape):

    inputs = Input(inputshape)
    x = Conv2D(50, (5,5), activation="relu")(inputs)
    x = MaxPooling2D()(x)
    
	
    x = Conv2D(100, (3, 3), activation="relu")(x)
    x = MaxPooling2D()(x)
    
    x = Conv2D(100, (3, 3), activation="relu")(x)
    
    x = Flatten()(x)
    outputs = Dense(2048, activation='sigmoid')(x)
    

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


def image_read(path):
  image = cv2.imread(path, 0)
  image = image.astype('float32')
  image /= 255.0
  image = cv2.resize(image, (120,120))
  image = image.reshape(120,120,1)

  return image



def siamese_datagen():
    pairs = [np.zeros(( 2*len(images), 120, 120, 1)) for i in range(2)]
    targets = np.array([1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0])
    for i in range(2*len(images)):
      print(2*len(images))
      for j in range(len(classes)):
        
        real_forged = os.path.join(base, classes[j])
        real_images = os.path.join(real_forged, 'real')
        forged_images = os.path.join(real_forged, 'forged') 

        for real in glob.glob(os.path.join(real_images, '*.png')):

          img1 = image_read(real)
          pairs[0][i,:,:,:]  = img1

          random_img = glob.glob(os.path.join(real_images, '*.png'))
          
          path = secrets.choice(random_img)
          img2 = image_read(path)
          
          pairs[1][i,:,:,:]  = img2
          ##targets.append('1')

          random_forged_img = glob.glob(os.path.join(forged_images, '*.png'))
          path2 = secrets.choice(random_forged_img)
          
          img2 = image_read(path2)

          pairs[0][i,:,:,:] = img1
          pairs[1][i,:,:,:] = img2
          
          ##targets.append('0')
          print('targets')
          print(targets)




    
    
    
 
  
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
 	inputs, targets,
 	batch_size=32, 
 	epochs=150)
    
        
