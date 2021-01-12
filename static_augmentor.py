import cv2
import numpy as np
import tensorflow as tf
from time import time

save_path = r''
img_channel = 1
augmentation_count = 1000


def get_original_image_paths():
    from glob import glob
    return glob(r'*.jpg')


def augment():
    global save_path, augmentation_count, img_channel
    generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.12,
        height_shift_range=0.12,
        brightness_range=(0.75, 1.25),
        shear_range=0.1,
        zoom_range=0.15)
    image_paths = get_original_image_paths()
    save_count = 0
    while True:
        for path in image_paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if img_channel == 1 else cv2.IMREAD_COLOR)
            height, width = img.shape[0], img.shape[1]
            x = np.asarray(img).reshape((1, height, width, img_channel))
            x = generator.flow(x=x, batch_size=1)[0][0]
            x = np.asarray(x).astype('uint8')
            cv2.imwrite(rf'{save_path}\generated_{int(time() / 1e-5)}{save_count}.jpg', x)
            save_count += 1
            print(f'save count : {save_count}')
            if save_count == augmentation_count:
                break
        if save_count == augmentation_count:
            break


if __name__ == '__main__':
    augment()
