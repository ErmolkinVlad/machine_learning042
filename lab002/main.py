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


def extract_dataset():
    with open('../data/notMNIST_sanit.pickle', 'rb') as f:
        data = pickle.load(f)
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

#1: Реализуйте полносвязную нейронную сеть с помощью библиотеки Tensor Flow.
# В качестве алгоритма оптимизации можно использовать, например,
# стохастический градиент (Stochastic Gradient Descent, SGD).
# Определите количество скрытых слоев от 1 до 5,
# количество нейронов в каждом из слоев до нескольких сотен,
# а также их функции активации (кусочно-линейная, сигмоидная, гиперболический тангенс и т.д.).

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(image_name(train_labels[i]))
# plt.show()


# Flatten преобразует формат изображения из двумерного массива (28 на 28 пикселей)
# в одномерный (размерностью 28 * 28 = 784 пикселя)
# Dense - это полносвязные нейронные слои.
baseline_model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation='sigmoid'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

baseline_model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy', 'sparse_categorical_crossentropy'])

baseline_model.summary()

baseline_history = baseline_model.fit(train_images,
                                      train_labels,
                                      epochs=10,
                                      validation_data=(valid_images, valid_labels))

test_loss, test_acc, _ = baseline_model.evaluate(test_images,  test_labels, verbose=2)

print('\nТочность на проверочных данных:', test_acc)

# 3: Используйте регуляризацию и метод сброса нейронов (dropout) для борьбы с переобучением.
# Как улучшилось качество классификации?

#3.1 Регуляризация
# https://www.tensorflow.org/tutorials/keras/overfit_and_underfit#add_weight_regularization

l2_regularization = 1e-4

l2_model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation='sigmoid', kernel_regularizer=regularizers.l2(l2_regularization)),
    keras.layers.Dense(100, activation='relu', kernel_regularizer=regularizers.l2(l2_regularization)),
    keras.layers.Dense(10, activation='softmax')
])

l2_model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy', 'sparse_categorical_crossentropy'])

l2_model_history = l2_model.fit(train_images,
                                train_labels,
                                epochs=10,
                                validation_data=(valid_images, valid_labels))

test_loss, test_acc, _ = l2_model.evaluate(test_images,  test_labels, verbose=2)

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

# Регуляризация показала худшие результаты, чем исходная модель (0.9094037 против 0.9144495).

# 3.2 метод сброса нейронов

dropout_model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation='sigmoid'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])

dropout_model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy', 'sparse_categorical_crossentropy'])

dropout_model_history = dropout_model.fit(train_images,
                                train_labels,
                                epochs=10,
                                validation_data=(valid_images, valid_labels))

test_loss, test_acc, _ = dropout_model.evaluate(test_images,  test_labels, verbose=2)

print('\nТочность на проверочных данных:', test_acc)

# Точность на проверочных данных: (0.890367 против 0.9144495 у исходной модели).

# 3.3 регуляризация + дропаут

l2_dropout_model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation='sigmoid', kernel_regularizer=regularizers.l2(l2_regularization)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(100, activation='relu', kernel_regularizer=regularizers.l2(l2_regularization)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])

l2_dropout_model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy', 'sparse_categorical_crossentropy'])

l2_dropout_model_history = l2_dropout_model.fit(train_images,
                                train_labels,
                                epochs=10,
                                validation_data=(valid_images, valid_labels))

test_loss, test_acc, _ = l2_dropout_model.evaluate(test_images,  test_labels, verbose=2)

print('\nТочность на проверочных данных:', test_acc)



plot_history(
    [
        ('baseline', baseline_history),
        ('l2', l2_model_history),
        ('dropout', dropout_model_history),
        ('l2 + dropout', l2_dropout_model_history)
    ],
    key='sparse_categorical_crossentropy')

# В итоге, дропаут только ухудшил результат модели,
# регуляризация её не ухудшила, но и лучших результатов не показала.
# Можно сделать вывод, что так как эти методы
# применяются для борьбы с переобучением модели,
# то если они ничего не улучшают,
# значит модель изначально не была переобучена.

# Задание 4.
# Воспользуйтесь динамически изменяемой скоростью обучения (learning rate).
# Наилучшая точность, достигнутая с помощью данной модели составляет 97.1%.
# Какую точность демонстрирует Ваша реализованная модель?

# Adagrad

# Adagrad выполняет большие обновления для более разреженных параметров
# и меньшие обновления для менее разреженных параметров.
# Он имеет хорошую производительность с разреженными данными
# и обучением крупномасштабной нейронной сети.
# Тем не менее, его монотонная скорость обучения
# обычно оказывается слишком агрессивной
# и перестает учиться слишком рано
# при обучении глубоких нейронных сетей.

adagrad_model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation='sigmoid'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

optimizer = keras.optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
adagrad_model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy', 'sparse_categorical_crossentropy'])

adagrad_model.summary()

adagrad_history = adagrad_model.fit(train_images,
                                      train_labels,
                                      epochs=10,
                                      validation_data=(valid_images, valid_labels))

test_loss, test_acc, _ = adagrad_model.evaluate(test_images,  test_labels, verbose=2)

print('\nТочность на проверочных данных:', test_acc)

# Adadelta

# Adadelta - это расширение Adagrad,
# которое стремится уменьшить свою агрессивную,
# монотонно уменьшающуюся скорость обучения.

adadelta_model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation='sigmoid'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

optimizer = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
adadelta_model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy', 'sparse_categorical_crossentropy'])

adadelta_model.summary()

adadelta_history = adadelta_model.fit(train_images,
                                      train_labels,
                                      epochs=10,
                                      validation_data=(valid_images, valid_labels))

test_loss, test_acc, _ = adadelta_model.evaluate(test_images,  test_labels, verbose=2)

print('\nТочность на проверочных данных:', test_acc)


# RMSprop

# RMSprop очень просто настраивает метод Адаграда,
# пытаясь уменьшить его агрессивное, монотонно убывающее обучение.

rms_prop_model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation='sigmoid'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
rms_prop_model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy', 'sparse_categorical_crossentropy'])

rms_prop_model.summary()

rms_prop_history = rms_prop_model.fit(train_images,
                                      train_labels,
                                      epochs=10,
                                      validation_data=(valid_images, valid_labels))

test_loss, test_acc, _ = rms_prop_model.evaluate(test_images,  test_labels, verbose=2)

print('\nТочность на проверочных данных:', test_acc)

# Adam

# Adam - это обновление оптимизатора RMSProp,
# похожее на RMSprop с динамикой.

adam_model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation='sigmoid'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
adam_model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy', 'sparse_categorical_crossentropy'])

adam_model.summary()

adam_history = adam_model.fit(train_images,
                                      train_labels,
                                      epochs=10,
                                      validation_data=(valid_images, valid_labels))

test_loss, test_acc, _ = adam_model.evaluate(test_images,  test_labels, verbose=2)

print('\nТочность на проверочных данных:', test_acc)