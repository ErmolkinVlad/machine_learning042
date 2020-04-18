from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow и tf.keras
import tensorflow as tf
from tensorflow import keras

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
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(242, activation='sigmoid'),
    keras.layers.Dense(242, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=3)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nТочность на проверочных данных:', test_acc)

predictions = model.predict(test_images)

np.argmax(predictions[0])

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(image_name(predicted_label),
                                100*np.max(predictions_array),
                                image_name(true_label)),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    # plt.grid(False)
    plt.grid(axis = 'y')
    plt.xticks(np.arange(10), [image_name(x) for x in np.arange(10)])
    # plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()