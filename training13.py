'''
By: William Zhi 
Demo of a two step Transfer learning.
Freeze initial layers in first run, then fine-tune the remaining network.
We are using VGGNet in this demo. The Keras and tensorflow libraries are required
'''
#ex1
import os
import csv
import numpy as np
import math

import tensorflow as tf

#used to fix bug
tf.python.control_flow_ops = tf

import keras

from keras import backend as K
#from keras.utils.generic_utils import get_from_module

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout
import param as p


from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K



def mean_absolute_error(y_true, y_pred):
    '''Calculates the mean absolute error (mae) rate
    between predicted and target values.
    '''
    trueV=tf.cast(K.argmax(y_true, axis=-1),tf.float32)
    predictedV=tf.cast(K.argmax(y_pred, axis=-1),tf.float32)
    return K.mean(K.abs(trueV-predictedV))

def count_samples(rootdir):
  
  totalcount=0
  for label in range(0,2):
    currentdir=rootdir+str(label)+'/'
    files=os.listdir(currentdir)
    totalcount+=len(files)
  return totalcount

def top_k_categorical_accuracy(y_true, y_pred, k=2):
    '''Calculates the top-k categorical accuracy rate, i.e. success when the
    target class is within the top-k predictions provided.
    '''
    return K.mean(tf.nn.in_top_k(y_pred, K.argmax(y_true, axis=-1), k))


def run_training():
  
  traindir=p.splitdir+'train/'
  testdir=p.splitdir+'test/'
  nTrainSample=count_samples(traindir)
  nTestSample=count_samples(testdir)
  img_width=224
  img_height=224

  base_model = VGG16(weights='imagenet', include_top=False)
    
  base_model.layers.pop()#layer18
  
  base_model.layers.pop()#layer17

  base_model.layers.pop()#layer16
  
  base_model.layers.pop()#layer15
  
  base_model.layers.pop()#layer14
  
  base_model.layers.pop()#layer13
  
  base_model.layers.pop()#layer12

  base_model.layers.pop()#layer11

  base_model.layers.pop()#layer10
  '''
  base_model.layers.pop()#layer9

  base_model.layers.pop()#layer8

  base_model.layers.pop()#layer7
  '''


  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(256, activation='relu')(x)
  x = Dropout(0.5)(x)
  predictions = Dense(2, activation='sigmoid')(x)
  model = Model(input=base_model.input, output=predictions)
  for layer in base_model.layers:
    layer.trainable = False

  # compile the model (should be done *after* setting layers to non-trainable)
  model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy', mean_absolute_error,top_k_categorical_accuracy])

  train_datagen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.2,
	horizontal_flip=True,
	vertical_flip=True)

  test_datagen = ImageDataGenerator(
        rescale=1./255)

  train_generator = train_datagen.flow_from_directory(
        traindir,
        target_size=(img_width, img_height),
        batch_size=64,
	class_mode='categorical')

  test_generator = test_datagen.flow_from_directory(
        testdir,
        target_size=(img_width, img_height),
        batch_size=64,
	class_mode='categorical')

  model.fit_generator(
        train_generator,
        samples_per_epoch=nTrainSample,
        nb_epoch=30,
        validation_data=test_generator,
        nb_val_samples=nTestSample)


  for i, layer in enumerate(base_model.layers):
    if layer.trainable==False:
      print(i, layer.name)
    layer.trainable = True



  # we need to recompile the model for these modifications to take effect
  # we use SGD with a low learning rate
  from keras.optimizers import SGD
  model.compile(optimizer=SGD(lr=0.00001, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy', mean_absolute_error,top_k_categorical_accuracy])


  train_datagen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.2,
	horizontal_flip=True,
	vertical_flip=True)

  test_datagen = ImageDataGenerator(
        rescale=1./255)

  train_generator = train_datagen.flow_from_directory(
        traindir,
        target_size=(img_width, img_height),
        batch_size=64,
	class_mode='categorical')

  test_generator = test_datagen.flow_from_directory(
        testdir,
        target_size=(img_width, img_height),
        batch_size=64,
	class_mode='categorical')

  model.fit_generator(
        train_generator,
        samples_per_epoch=nTrainSample,
        nb_epoch=30,
        validation_data=test_generator,
        nb_val_samples=nTestSample)


  #save files
  #model_json = model.to_json()

  model.save_weights('./netlayersremoved400_1.h5')
