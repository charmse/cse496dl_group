import os
import util
import numpy as np
import pandas as pd
import tensorflow as tf
import itertools as itr
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.misc import imread
from PIL import Image
from random import randrange
from collections import Counter
#%matplotlib inline

#cleverhans
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval
from cleverhans.utils import AccuracyReport
from cleverhans.utils_keras import cnn_model, KerasModelWrapper
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import FastGradientMethod, LBFGS, BasicIterativeMethod
from cleverhans.utils import AccuracyReport

#keras
import keras
from keras import __version__
from keras import backend as K
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.datasets import mnist

def make(x,struct):
    return x

def add_new_last_layer(base_model, nb_classes):
  """Add last layer to the convnet
  Args:
    base_model: keras model excluding top
    nb_classes: # of classes
  Returns:
    new keras model with last layer
  """
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(1024, activation='relu')(x) 
  predictions = Dense(nb_classes, activation='softmax')(x) 
  model = Model(input=base_model.input, output=predictions)
  return model

def setup_to_finetune(model):
   """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top 
      layers.
   note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in 
         the inceptionv3 architecture
   Args:
     model: keras model
   """
   for layer in model.layers:
      layer.trainable = True
   model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),   
                 loss='categorical_crossentropy')