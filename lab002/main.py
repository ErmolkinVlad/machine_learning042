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


def pickle_dataset(pickle_name, train_dataset, train_labels, test_dataset, test_labels):
    try:
        f = open(pickle_name, 'wb')
        save = [(train_dataset, train_labels ), (test_dataset, test_labels) ]
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_name, ':', e)
        raise

def extract_dataset(pickle_name):
    with open(pickle_name, 'rb') as f:
        data = pickle.load(f)
    return data

def load_letter(folder):
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), 28, 28), dtype=np.float32)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            image_data = (ndimage.imread(image_file).astype(float) - pixel_depth / 2) / pixel_depth
            if image_data.shape != (28, 28):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]
    return dataset

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

def load_dataset():
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    letters_datasets = [(letter, load_letter(letter)) for letter in letters]
    num_of_images = np.sum([len(letter_dataset) for (_, letter_dataset) in letters_datasets])
    dataset = np.ndarray((num_of_images, 28, 28), dtype=np.float32)
    labels = np.ndarray(num_of_images, dtype=np.int32)
    for (letter, dataset) in letters_datasets:
        dataset.append(dataset)
        labels.append(np.full(len(dataset), letter))
    
    return labels, dataset

# def merge_datasets(datasets, train_size):
#     valid_dataset, valid_labels = make_arrays(valid_size, image_size)
#     train_dataset, train_labels = make_arrays(train_size, image_size)
#     vsize_per_class = valid_size // num_classes
#     tsize_per_class = train_size // num_classes

#     start_v, start_t = 0, 0
#     end_v, end_t = vsize_per_class, tsize_per_class
#     end_l = vsize_per_class + tsize_per_class
#     for label, pickle_file in enumerate(pickle_files):
#         try:
#             with open(pickle_file, 'rb') as f:
#                 letter_set = pickle.load(f)
#                 # let's shuffle the letters to have random validation and training set
#                 np.random.shuffle(letter_set)
#                 if valid_dataset is not None:
#                     valid_letter = letter_set[:vsize_per_class, :, :]
#                     valid_dataset[start_v:end_v, :, :] = valid_letter
#                     valid_labels[start_v:end_v] = label
#                     start_v += vsize_per_class
#                     end_v += vsize_per_class

#                 train_letter = letter_set[vsize_per_class:end_l, :, :]
#                 train_dataset[start_t:end_t, :, :] = train_letter
#                 train_labels[start_t:end_t] = label
#                 start_t += tsize_per_class
#                 end_t += tsize_per_class
#         except Exception as e:
#             print('Unable to process data from', pickle_file, ':', e)
#             raise

#     return valid_dataset, valid_labels, train_dataset, train_labels

pickle_name = 'archive.pickle'
if os.path.exists(pickle_name):
    print('exist')
    data = extract_dataset(pickle_name)
else:
    print('not exist')
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    pickle_dataset(pickle_name,
        train_dataset=train_images,
        train_labels=train_labels,
        test_dataset=test_images,
        test_labels=test_labels)
    data = fashion_mnist.load_data()

(train_images, train_labels), (test_images, test_labels) = data
pdb.set_trace()
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

train_images = (train_images - 255.0 / 2)/ 255.0
test_images = (test_images - 255.0 / 2) / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
# plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
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

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    # plt.grid(False)
    plt.grid(axis = 'y')
    plt.xticks(np.arange(10))
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