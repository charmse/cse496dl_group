import os
import util
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
print("Finished Import")

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'directory where MNIST is located')
flags.DEFINE_string('save_dir', '', 'directory where model graph and weights are saved')
flags.DEFINE_integer('batch_size', 16, '')
flags.DEFINE_float('lr', 0.001, '')
flags.DEFINE_integer('early_stop', 20, '')
flags.DEFINE_string('db', 'emodb', '')
flags.DEFINE_integer('epochs', 100, '')
flags.DEFINE_float('reg_coeff', 0.001, '')
flags.DEFINE_float('split', 0.90, '')
flags.DEFINE_string('master', '', 'The address of the TensorFlow master to use.')
flags.DEFINE_string('checkpoint_path', 'nips-2017-adversarial-learning-development-set/inception_v3.ckpt', 'Path to checkpoint for inception network.')
flags.DEFINE_string('input_dir', 'nips-2017-adversarial-learning-development-set/images/', 'Input directory with images.')
flags.DEFINE_string('output_dir', '', 'Output directory with images.')
flags.DEFINE_float('max_epsilon', 4.0, 'Maximum size of adversarial perturbation.')
flags.DEFINE_integer('image_width', 299, 'Width of each input images.')
flags.DEFINE_integer('image_height', 299, 'Height of each input images.')
flags.DEFINE_float('eps', 2.0 * 16.0 / 255.0, '')
flags.DEFINE_integer('num_classes', 1001, '')
FLAGS = flags.FLAGS

def main(argv):

    print("Start Main")
    # Set arguments:  Save_Dir Structure Learning_Rate Earling_Stoping Batch_Size Data_Dir    
    data_dir = FLAGS.data_dir
    save_dir = FLAGS.save_dir
    learning_rate = FLAGS.lr
    early_stop = FLAGS.early_stop
    batch_size = FLAGS.batch_size
    reg_coeff = FLAGS.reg_coeff
    split = FLAGS.split
    master = FLAGS.master
    checkpoint_path = FLAGS.checkpoint_path
    input_dir = FLAGS.input_dir
    output_dir = FLAGS.output_dir
    image_width = FLAGS.image_width
    image_height = FLAGS.eps
    num_classes = FLAGS.num_classes
    eps = FLAGS.eps
    batch_shape = [batch_size, image_height, image_width, 3]

    tf.logging.set_verbosity(tf.logging.INFO)

    #load training data
    imgs,labels,names = util.load_training_images('tiny-imagenet-200/train/')
    print("Training Images Loaded")
    
    #retrype and resize training data
    imgs = imgs[0:100]
    labels = labels[0:100]
    names = names[0:100]
    imgs_large = np.ndarray(shape= [imgs.shape[0],299,299,3])
    for i in range(imgs.shape[0]):
        imgs_large[i,:,:,:] = util.rescale(imgs[i])
    imgs_large=imgs_large.astype('uint8')
    imgs_noisy = np.ndarray(shape= imgs_large.shape)
    for i in range(imgs_large.shape[0]):
        imgs_noisy[i,:,:,:] = util.noisy(1,imgs_large[i])
    imgs_noisy=imgs_noisy.astype('uint8')
    sub_imgs,sub_labels = util.subsample(imgs_noisy,labels)
    batch_shape = [20, 299, 299, 3]
    num_classes = 200

    base_model = InceptionV3(weights='imagenet', include_top=False)
    model = model.add_new_last_layer(base_model, 201)



class InceptionModel(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False

  def __call__(self, x_input):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    with slim.arg_scope(inception.inception_v3_arg_scope()):
      _, end_points = inception.inception_v3(
          x_input, num_classes=self.num_classes, is_training=False,
          reuse=reuse)
    self.built = True
    output = end_points['Predictions']
    # Strip off the extra reshape op at the output
    probs = output.op.inputs[0]
    return probs

if __name__ == "__main__":
    tf.app.run()