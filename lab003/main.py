from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow и tf.keras
import tensorflow as tf
from tensorflow import keras
from keras import regularizers

# Вспомогательные библиотеки
import numpy as np
import matplotlib.pyplot as plt
import pdb
from six.moves import cPickle as pickle
import os
from scipy import ndimage

# Useful links:
# https://habr.com/ru/post/309508/1
# https://xn--90aeniddllys.xn--p1ai/svertochnaya-nejronnaya-set-na-python-i-keres/

def extract_dataset():
    with open('../data/notMNIST_sanit.pickle', 'rb') as f:
        data = pickle.load(f)

    # reshape dataset because of error:
    # ValueError: Error when checking input: expected conv2d_input to have 4 dimensions,
    # but got array with shape (200000, 28, 28)
    for key, dataset in data.items():
        data[key] = dataset.reshape(*dataset.shape, 1)
    return data

def image_name(index):
  return chr(ord('A') + index)

dataset = extract_dataset()
train_images = dataset['train_dataset']
train_labels = dataset['train_labels']
valid_images = dataset['valid_dataset']
valid_labels = dataset['valid_labels']
test_images = dataset['test_dataset']
test_labels = dataset['test_labels']

# Задание 1. Реализуйте нейронную сеть с двумя сверточными слоями,
# и одним полносвязным с нейронами с кусочно-линейной функцией активации.
# Какова точность построенное модели?

# kernel_size=3 — размер ядра 3х3.
# Функция активации 'relu' ( Rectified Linear Activation ),
# 64 это число ядер свертки( сколько признаком будем искать).
# Flatten() – слой, преобразующий 2D-данные в 1D-данные.
conv2d_model = keras.Sequential([
    keras.layers.Conv2D(8, kernel_size=3, activation='relu', input_shape=(28, 28, 1)),
    keras.layers.Conv2D(16, kernel_size=3, activation='relu', input_shape=(28, 28, 1)),
    keras.layers.Flatten(),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

conv2d_model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

conv2d_model.summary()

conv2d_history = conv2d_model.fit(train_images,
                                      train_labels,
                                      epochs=10,
                                      validation_data=(valid_images, valid_labels))

test_loss, test_acc = conv2d_model.evaluate(test_images,  test_labels, verbose=2)

print('\nТочность на проверочных данных:', test_acc)

# Задание 2.
# Замените один из сверточных слоев на слой,
# реализующий операцию пулинга (Pooling) с функцией максимума или среднего.
# Как это повлияло на точность классификатора?

# Helpful links:
# https://neurohive.io/ru/osnovy-data-science/glubokaya-svertochnaja-nejronnaja-set/
# https://medium.com/@congyuzhou/pooling-%D1%81%D0%BB%D0%BE%D0%B9-dbe00ef48eab

maxpool_model = keras.Sequential([
    keras.layers.Conv2D(8, kernel_size=3, activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

maxpool_model.compile(optimizer='sgd',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

maxpool_model.summary()

maxpool_history = maxpool_model.fit(train_images,
                                    train_labels,
                                    epochs=10,
                                    validation_data=(valid_images, valid_labels))

test_loss, test_acc = maxpool_model.evaluate(test_images,  test_labels, verbose=2)

print('\nТочность на проверочных данных:', test_acc)

# Задание 3.
# Реализуйте классическую архитектуру сверточных сетей LeNet-5 (http://yann.lecun.com/exdb/lenet/).


# padding: https://medium.com/@congyuzhou/padding-32266fa95816
# https://medium.com/@congyuzhou/lenet-5-%D1%81%D0%B2%D0%BE%D0%B8%D0%BC%D0%B8-%D1%80%D1%83%D0%BA%D0%B0%D0%BC%D0%B8-b60ae3727cd3

# stride - шаг
lenet5_model = keras.Sequential([
    keras.layers.Conv2D(filters=6, kernel_size=[5,5], padding='same', activation='relu'),
    keras.layers.MaxPooling2D(pool_size=[2,2], strides=2),
    keras.layers.Conv2D(filters=16, kernel_size=[5,5], padding='valid', activation='relu'),
    keras.layers.MaxPooling2D(pool_size=[2,2], strides=2),
    keras.layers.Flatten(),
    keras.layers.Dense(units=120, activation='relu'),
    keras.layers.Dense(units=84, activation='relu'),
    keras.layers.Dense(units=10, activation = 'softmax')
])

lenet5_model.compile(optimizer='sgd',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

lenet5_model.summary()

lenet5_history = lenet5_model.fit(train_images,
                                    train_labels,
                                    epochs=10,
                                    validation_data=(valid_images, valid_labels))

test_loss, test_acc = lenet5_model.evaluate(test_images,  test_labels, verbose=2)

print('\nТочность на проверочных данных:', test_acc)


def plot_history(histories, key='binary_crossentropy'):
    plt.figure(figsize=(16,10))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_' + key],
                        '--', label=name.title()+' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()

    plt.xlim([0,max(history.epoch)])
    
    plt.show()

plot_history(
    [
        ('conv2d', conv2d_history),
        ('maxpool', maxpool_history),
        ('lenet5', lenet5_history)
    ],
    key='loss')