import cv2
import os
from matplotlib import pyplot as plt
import random

# 1: load data and show some images

def load_images_from_folder(folder):
    images = []
    for name in os.listdir(folder):
        is_folder = os.path.isdir(os.path.join(folder, name))
        if is_folder:
            subfolder = os.path.join(folder, name)
            images += load_images_from_folder(subfolder)
        else:
            img = cv2.imread(os.path.join(folder, name))
            if img is not None:
                images.append(img)
    return images

def show_image(image):
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.imshow(image, cmap = 'gray', interpolation = 'bicubic')
    plt.show()

img_dir_small = './notMNIST_small'
img_dir_large = './notMNIST_large'

images_small = load_images_from_folder(img_dir_small)
images_large = load_images_from_folder(img_dir_large)

for image in random.choices(images_small, k=10):
    show_image(image)