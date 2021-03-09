import os
from glob import glob
from time import time

import cv2
import numpy as np
import tensorflow as tf

img_channel = 1
target_num_images = 4000


def augment(image_path):
    global target_num_images, img_channel
    image_path = image_path.replace('\\', '/')
    generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.02,
        height_shift_range=0.02,
        brightness_range=(0.7, 1.3),
        shear_range=0.05,
        zoom_range=0.05)
    image_paths = glob(f'{image_path}/*.jpg')
    image_count = len(image_paths)
    if image_count < target_num_images:
        while True:
            for path in image_paths:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if img_channel == 1 else cv2.IMREAD_COLOR)
                height, width = img.shape[0], img.shape[1]
                x = np.asarray(img).reshape((1, height, width, img_channel))
                x = generator.flow(x=x, batch_size=1)[0][0]
                x = np.asarray(x).astype('uint8')
                cv2.imwrite(rf'{image_path}\generated_{int(time() / 1e-5)}{image_count}.jpg', x)
                image_count += 1
                print(f'image count : {image_count}')
                if image_count == target_num_images:
                    return


if __name__ == '__main__':
    for dir_path in glob('*'):
        if os.path.isdir(dir_path):
            augment(dir_path)
