import os
import model
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

def split_data(data, labels, proportion):
    """                                                                                                                                                                                                                                                                                                 
    """
    size = data.shape[0]
    np.random.seed(42)
    indicies = np.random.permutation(size)
    split_idx = int(proportion * size)
    return data[indicies[:split_idx]], data[indicies[split_idx:]], labels[indicies[:split_idx]], labels[indicies[split_idx:]]

def transform_labels(labels):
    new_labels = []
    for l in labels:
        t_label = np.zeros(10)
        t_label[(int(l)-1)] = 1.
        new_labels.append(t_label)
    return np.array(new_labels)

def onehot(data):
    data_final = np.zeros((data.shape[0],10))
    data_final[np.arange(data.shape[0]),data.astype(int)]=1
    return data_final

def noisy(noise_typ,image):
    #Gaussian
   if noise_typ == 1:
      row,col,ch= image.shape
      mean = 0
      var = 0.1
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy.astype('uint8')
    #Salt and Pepper
   elif noise_typ == 2:
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1
      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out
    #Poisson
   elif noise_typ == 3:
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
    #speckle
   elif noise_typ ==4:
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)        
      noisy = image + image * gauss
      return noisy.astype('uint8')

def psnr(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    psnr = 20*np.log10(255) - 10*np.log10(err)
    return psnr

def subsample(training_images,training_labels, ratio=0.8):
    shuffle_index = np.random.permutation(len(training_labels))
    training_images = training_images[shuffle_index]
    training_labels = training_labels[shuffle_index]
    sample = list()
    sample_labels = list()
    n_sample = round(training_images.shape[0] * ratio)
    while len(sample) < n_sample:
        index = randrange(training_images.shape[0])
        sample.append(training_images[index,:,:,:])
        sample_labels.append(training_labels[index])
    return np.asarray(sample),np.asarray(sample_labels)

# def load_images(input_dir):
#     filenames = []
#     idx = 0
#     #batch_size = batch_shape[0]
#     # Limit to first 20 images for this example
#     for filepath in sorted(tf.gfile.Glob(os.path.join(input_dir, '*.png'))):
#         filenames.append(os.path.basename(filepath))
#     x = np.array([np.array(Image.open(os.path.join(input_dir, fname))) for fname in filenames])
#     return x

def load_images(input_dir, batch_shape):
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    # Limit to first 20 images for this example
    for filepath in sorted(tf.gfile.Glob(os.path.join(input_dir, '*.png')))[:20]:
        with tf.gfile.Open(filepath, "rb") as f:
            images[idx, :, :, :] = imread(f, mode='RGB').astype(np.float)*2.0/255.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images

def load_training_images(training_image_dir):

    image_index = 0
    
    images = np.ndarray(shape=(500*200, 64,64,3))
    names = []
    labels = []                       
    
    # Loop through all the types directories
    for type in os.listdir(training_image_dir):
        if os.path.isdir(training_image_dir + type + '/images/'):
            type_images = os.listdir(training_image_dir + type + '/images/')
            # Loop through all the images of a type directory
            #batch_index = 0;
            #print ("Loading Class ", type)
            for image in type_images:
                image_file = os.path.join(training_image_dir, type + '/images/', image)

                # reading the images as they are; no normalization, no color editing
                image_data = mpimg.imread(image_file) 
                #print ('Loaded Image', image_file, image_data.shape)
                if (image_data.shape == (64, 64, 3)):
                    images[image_index, :,:,:] = image_data
                    
                    labels.append(type)
                    names.append(image)
                    
                    image_index += 1
                    #batch_index += 1
                #if (batch_index >= batch_size):
                 #   break;
    labels = np.asarray(labels)
    names = np.asarray(names)
    return (images[0:len(labels)].astype('uint8'), labels, names)

def save_images(images, filenames, output_dir):
    for i, filename in enumerate(filenames):
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
            Image.fromarray(img).save(f, format='PNG')