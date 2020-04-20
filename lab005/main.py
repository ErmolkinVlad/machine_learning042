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
    keras.layers.Conv2D(32, 3, activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.1),

    keras.layers.Conv2D(64, 3, activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.1),

    keras.layers.Conv2D(128, 3, activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.1),

    keras.layers.Conv2D(256, 3, activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.1),

    keras.layers.Flatten(),

    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])


basic_model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

basic_model.summary()

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


# Задание 3.
# Примените дополнение данных (data augmentation). Как это повлияло на качество классификатора?

aug_data_generator = ImageDataGenerator(rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')


aug_train_iterator = aug_data_generator.flow_from_directory(train_folder, class_mode='binary', target_size=(IMAGE_SIZE, IMAGE_SIZE), shuffle=True)
aug_valid_iterator = aug_data_generator.flow_from_directory(valid_folder, class_mode='binary', target_size=(IMAGE_SIZE, IMAGE_SIZE), shuffle=True)
aug_test_iterator = aug_data_generator.flow_from_directory(test_folder, class_mode='binary', target_size=(IMAGE_SIZE, IMAGE_SIZE), shuffle=True)


aug_model = keras.Sequential([
    keras.layers.Conv2D(32, 3, activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.1),
    
    keras.layers.Conv2D(64, 3, activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.1),
    
    keras.layers.Conv2D(128, 3, activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.1),
    
    keras.layers.Conv2D(256, 3, activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.1),
    
    keras.layers.Flatten(),
    
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

aug_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
aug_model.summary()
aug_model_history = aug_model.fit_generator(aug_train_iterator, steps_per_epoch=150, epochs=10, validation_data=aug_valid_iterator, validation_steps=50, verbose=1)

plot_history([('basic model', basic_model_history)], key='loss')


# 4

image_size = 224
input_shape = (224, 224, 3)

pre_trained_model = keras.applications.VGG19(input_shape=input_shape, include_top=False, weights="imagenet")

for i, layer in enumerate(pre_trained_model.layers):
    if i <= 42:
        layer.trainable = False
    else:
        layer.trainable = True

last_layer = pre_trained_model.get_layer('block5_pool')
last_output = last_layer.output

model = keras.layers.GlobalAveragePooling2D()(last_output)
model = keras.layers.Dense(512, activation='relu')(model)
model = keras.layers.Dropout(0.5)(model)
model = keras.layers.Dense(1, activation='sigmoid')(model)

vgg_model = keras.models.Model(pre_trained_model.input, model)

vgg_model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

vgg_model.summary()