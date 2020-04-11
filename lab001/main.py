import cv2
import os
from matplotlib import pyplot as plt
import random
import pdb

# 1: load data and show some images

# Returns data in format { 'folders': {}, 'images': { 'name': file, ...} }
def load_images_from_folder(folder):
    print("Getting images from {}".format(folder))
    result = { 'folders': {}, 'images': {} }
    for name in os.listdir(folder):
        is_folder = os.path.isdir(os.path.join(folder, name))
        if is_folder:
            subfolder = os.path.join(folder, name)
            result['folders'][name] = load_images_from_folder(subfolder)
        else:
            img = cv2.imread(os.path.join(folder, name))
            if img is not None:
                result['images'][name] = img
    return result

def plot_samples(images, sample_size, name):
    figure = plt.figure()
    figure.suptitle(name)
    folders = images['folders']
    for folder_name, folder in folders.items():
        image_name_samples = random.sample(list(folder['images']), sample_size)
        for image_name in image_name_samples:
            subplot = figure.add_subplot(
                sample_size,
                len(folders),
                list(folders).index(folder_name) * sample_size + image_name_samples.index(image_name) + 1
            )
            subplot.imshow(folder['images'][image_name])
            subplot.set_axis_off()
    plt.show()

test_sample_folder = './notMNIST_small'
train_sample_folder = './notMNIST_large'

train_sample = load_images_from_folder(train_sample_folder)
test_sample = load_images_from_folder(test_sample_folder)
plot_samples(test_sample, 10, 'hello')
