# Лабораторная работа №5. Применение сверточных нейронных сетей (бинарная классификация)

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

IMAGE_SIZE = 150

# Задание 1.
# Загрузите данные. Разделите исходный набор данных на обучающую, валидационную и контрольную выборки.

# Useful links:
# 1) https://machinelearningmastery.com/how-to-load-large-datasets-from-directories-for-deep-learning-with-keras/

def extract_dataset(name):
    zip_path = os.path.join('../data/cats_vs_dogs/', name + '.zip')
    if not os.path.exists(os.path.join('../data/cats_vs_dogs/', name)):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("../data/cats_vs_dogs")

extract_dataset('train')

def get_file_list_from_dir(folder_path):
    all_files = os.listdir(folder_path)
    data_files = list(filter(lambda file: file.endswith('.jpg'), all_files))
    return data_files


filelist = get_file_list_from_dir('../data/cats_vs_dogs/train')

## Used only once
# image_files = os.listdir('../data/cats_vs_dogs/train')
# for i, file in enumerate(image_files):
#     if "jpg" not in file:
#         continue

#     animal, im_id, _ = file.split('.')
#     if i <= len(image_files) * 0.1:
#         subfolder = '../data/cats_vs_dogs/valid'
#     elif i <= len(image_files) * 0.1 + len(image_files) * 0.2:
#         subfolder = '../data/cats_vs_dogs/test'
#     else:
#         subfolder = '../data/cats_vs_dogs/train'

#     os.rename(os.path.join('../data/cats_vs_dogs/train',file), os.path.join(subfolder, animal + 's', file))

train_folder = '../data/cats_vs_dogs/train'
valid_folder = '../data/cats_vs_dogs/valid'
test_folder = '../data/cats_vs_dogs/test'

train_cats_len = len(os.listdir(os.path.join(train_folder, 'cats') ))
train_dogs_len = len(os.listdir(os.path.join(train_folder, 'dogs')))
valid_cats_len = len(os.listdir(os.path.join(valid_folder, 'cats') ))
valid_dogs_len = len(os.listdir(os.path.join(valid_folder, 'dogs') ))
test_cats_len = len(os.listdir(os.path.join(test_folder, 'cats')))
test_dogs_len = len(os.listdir(os.path.join(test_folder, 'dogs')))

print(train_cats_len, train_dogs_len, valid_cats_len, valid_dogs_len, test_cats_len, test_dogs_len)

# using rescale because of empty images issue (https://stackoverflow.com/questions/43292009/keras-and-imagegenerator-outputs-black-images)
data_generator = ImageDataGenerator(rescale=1./255)

train_iterator = data_generator.flow_from_directory(train_folder, class_mode='binary', target_size=(IMAGE_SIZE, IMAGE_SIZE), shuffle=True)
valid_iterator = data_generator.flow_from_directory(valid_folder, class_mode='binary', target_size=(IMAGE_SIZE, IMAGE_SIZE), shuffle=True)
test_iterator = data_generator.flow_from_directory(test_folder, class_mode='binary', target_size=(IMAGE_SIZE, IMAGE_SIZE), shuffle=True)


def plot_samples(iterator, sample_size, name):
    figure, axes = plt.subplots(1, 5, figsize=(20, 20))
    figure.suptitle(name)
    axes = axes.flatten()
    images = next(iterator)[0][:sample_size]
    for img, ax in zip(images, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# plot_samples(train_iterator, 5, 'train')
# plot_samples(valid_iterator, 5, 'test')
# plot_samples(test_iterator, 5, 'valid')

# Задание 2.
# Реализуйте глубокую нейронную сеть с как минимум тремя сверточными слоями. Какое качество классификации получено?

l2_regularization = 1e-4

basic_model = keras.Sequential([
    keras.layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Flatten(),
    keras.layers.Dense(units=512, activation='relu', kernel_regularizer=regularizers.l2(l2_regularization)),
    keras.layers.Dropout(0.25),

    keras.layers.Dense(units=10, activation = 'softmax')
])

basic_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

basic_model.summary()

basic_model_history = basic_model.fit(train_iterator, epochs=10, validation_data=valid_iterator)
test_loss, test_acc = basic_model.evaluate(test_iterator)

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

plot_history([('basic model, basic_model_history')], key='loss')