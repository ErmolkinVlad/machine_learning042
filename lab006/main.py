# Лабораторная работа №6. Применение сверточных нейронных сетей (многоклассовая классификация)

# TensorFlow и tf.keras
import tensorflow as tf
from tensorflow import keras
from keras import regularizers

import numpy as np
import matplotlib.pyplot as plt
import pdb
import os
import scipy.io
from sklearn.model_selection import train_test_split
import tarfile
from six.moves import cPickle as pickle
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Задание 1.
# Загрузите данные. Разделите исходный набор данных на обучающую и валидационную выборки.

def read_images(filename):
    images = []
    with open(filename) as f:
        f.readline()
        for line in f:
            label, *values = line.strip().split(',')
            image = np.array([float(v) for v in values]) / 255
            image.resize((28, 28, 1))
            images.append((int(label), image))

    labels = [p[0] for p in images]
    images = [p[1] for p in images]
    return np.array(labels), np.array(images)

test_labels, test_images = read_images("../data/sign_mnist/sign_mnist_test.csv")
train_labels, train_images = read_images("../data/sign_mnist/sign_mnist_train.csv")
train_images, valid_images, train_labels, valid_labels = train_test_split(train_images, train_labels)

print(len(train_images), len(train_labels), len(test_images), len(test_labels), len(valid_images), len(valid_labels))

def plot_samples(images, sample_size, name):
    figure, axes = plt.subplots(1, sample_size, figsize=(28, 28))
    figure.suptitle(name)
    axes = axes.flatten()
    for img, ax in zip(images, axes):
        img = np.array(img)
        img.resize((28, 28))
        axes.imshow(img, cmap='gray')
        axes.axis('off')
    plt.tight_layout()
    plt.show()
    
plot_samples(train_images, 5, 'train')
plot_samples(valid_images, 5, 'valid')
plot_samples(test_images, 5, 'test')

train_images_3d = np.array([np.repeat(img, 3, 2) for img in tf.image.resize(train_images, (32, 32))])
valid_images_3d = np.array([np.repeat(img, 3, 2) for img in tf.image.resize(valid_images, (32, 32))])
test_images_3d = np.array([np.repeat(img, 3, 2) for img in tf.image.resize(test_images, (32, 32))])
