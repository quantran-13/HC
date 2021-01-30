import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import cv2
import tensorflow as tf

from seg.config import config

# https://github.com/HasnainRaz/SemSegPipeline/blob/master/dataloader.py
AUTOTUNE = tf.data.experimental.AUTOTUNE


class DataLoader(object):
    """
        A TensorFlow Dataset API based loader for semantic segmentation problems.
    """

    def __init__(self, root, mode="train", augmentation=False, compose=False, one_hot_encoding=False, palette=None, image_size=(216, 320, 1)):
        """
        root: "./data/training_set"
        """
        super().__init__()
        self.root = root
        self.mode = mode
        self.augmentation = augmentation
        self.compose = compose
        self.one_hot_encoding = one_hot_encoding
        self.palette = palette
        self.image_size = (image_size[0], image_size[1])

        if (self.mode == "train"):
            self.df = pd.read_csv("./data/train.csv")
        elif (self.mode == "valid"):
            self.df = pd.read_csv("./data/valid.csv")
        elif (self.mode == "test"):
            import glob
            self.df = pd.read_csv("./data/test_set_pixel_size.csv")

        self.parse_data_path()

    def parse_data_path(self):
        if self.mode in ["train", "valid"]:
            self.image_paths = [os.path.join(self.root, _[0])
                                for _ in self.df.values.tolist()]

            self.mask_paths = [_.replace(".png", "_Annotation.png")
                               for _ in self.image_paths]
        elif self.mode == "test":
            self.image_paths = [os.path.join(self.root, _[0])
                                for _ in self.df.values.tolist()]

    def parse_data(self, image_paths, mask_paths=None):
        image_content = tf.io.read_file(image_paths)
        # grayscale
        images = tf.image.decode_png(image_content, channels=1)
        images = tf.cast(images, tf.float32)

        if self.mode in ["train", "valid"]:
            mask_content = tf.io.read_file(mask_paths)
            # grayscale
            masks = tf.image.decode_png(mask_content, channels=1)
            masks = tf.cast(masks, tf.float32)

            return images, masks

        return images

    def normalize_data(self, image, mask=None):
        """
            Normalizes image
        """
        image /= 255.
        # image = tf.image.per_image_standardization(image)

        if mask is not None:
            return image, mask

        return image

    def resize_data(self, image, mask=None):
        """
            Resizes image to specified size but still keep aspect ratio
        """
        image = tf.image.resize_with_pad(
            image, self.image_size[0], self.image_size[1], method="nearest")

        if mask is not None:
            mask = tf.image.resize_with_pad(
                mask, self.image_size[0], self.image_size[1], method="nearest")

            return image, mask

        return image

    def one_hot_encode(self, image, mask):
        """
            One hot encodes mask
        """
        one_hot_map = []

        for colour in self.palette:
            class_map = tf.reduce_all(tf.equal(mask, colour), axis=-1)
            one_hot_map.append(class_map)

        one_hot_map = tf.stack(one_hot_map, axis=-1)
        one_hot_map = tf.cast(one_hot_map, tf.float32)

        return image, one_hot_map

    def change_brightness(self, image, mask):
        """
            Randomly applies a random brightness change.
        """
        cond_brightness = tf.cast(tf.random.uniform(
            [], maxval=2, dtype=tf.int32), tf.bool)
        image = tf.cond(cond_brightness,
                        lambda: tf.image.random_brightness(image, 0.1),
                        lambda: tf.identity(image))

        return image, mask

    def change_contrast(self, image, mask):
        """
            Randomly applies a random contrast change.
        """
        cond_contrast = tf.cast(tf.random.uniform(
            [], maxval=2, dtype=tf.int32), tf.bool)
        image = tf.cond(cond_contrast,
                        lambda: tf.image.random_contrast(image, 0.1, 0.5),
                        lambda: tf.identity(image))

        return image, mask

    def flip_horizontally(self, image, mask):
        """
            Randomly flips image and mask horizontally in accord.
        """
        comb_tensor = tf.concat([image, mask], axis=2)
        comb_tensor = tf.image.random_flip_left_right(comb_tensor)
        image, mask = tf.split(comb_tensor, [1, 1], axis=2)

        return image, mask

    def reorder_channel(self, tensor, order="channel_first"):
        if order == "channel_first":
            return tf.convert_to_tensor(np.moveaxis(tensor.numpy(), -1, 0))
        else:
            return tf.convert_to_tensor(np.moveaxis(tensor.numpy(), 0, -1))

    def _tranform(self, tensor, types):
        tensor = self.reorder_channel(tensor)

        if types == "rotate":
            tensor = tf.keras.preprocessing.image.random_rotation(tensor, 15)
        elif types == "shift":
            tensor = tf.keras.preprocessing.image.random_shift(
                tensor, 0.1, 0.1)
        else:
            tensor = tf.keras.preprocessing.image.random_zoom(
                tensor, (1.2, 1.2))

        tensor = self.reorder_channel(
            tf.convert_to_tensor(tensor), order="channel_last")

        return tensor

    def rotate(self, image, mask):
        """
            Randomly rotates image and mask
        """
        cond_rotate = tf.cast(tf.random.uniform(
            [], maxval=2, dtype=tf.int32), tf.bool)
        comb_tensor = tf.concat([image, mask], axis=2)
        comb_tensor = tf.cond(cond_rotate,
                              lambda: self._tranform(comb_tensor, "rotate"),
                              lambda: tf.identity(comb_tensor))
        image, mask = tf.split(comb_tensor, [1, 1], axis=2)

        return image, mask

    def shift(self, image, mask):
        """
            Randomly translates image and mask
        """
        cond_shift = tf.cast(tf.random.uniform(
            [], maxval=2, dtype=tf.int32), tf.bool)
        comb_tensor = tf.concat([image, mask], axis=2)
        comb_tensor = tf.cond(cond_shift,
                              lambda: self._tranform(comb_tensor, "shift"),
                              lambda: tf.identity(comb_tensor))
        image, mask = tf.split(comb_tensor, [1, 1], axis=2)

        return image, mask

    def zoom(self, image, mask):
        """
            Randomly scale image and mask
        """
        cond_zoom = tf.cast(tf.random.uniform(
            [], maxval=2, dtype=tf.int32), tf.bool)
        comb_tensor = tf.concat([image, mask], axis=2)
        comb_tensor = tf.cond(cond_zoom,
                              lambda: self._tranform(comb_tensor, "zoom"),
                              lambda: tf.identity(comb_tensor))
        image, mask = tf.split(comb_tensor, [1, 1], axis=2)

        return image, mask

    def _equalize_histogram(self, image):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_his_eq = clahe.apply(image.numpy().astype(np.uint8))
        image = np.expand_dims(img_his_eq, axis=-1)

        return tf.convert_to_tensor(image, dtype=tf.float32)

    def equalize_histogram(self, image, mask):
        """
            Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        """
        cond_he = tf.cast(tf.random.uniform(
            [], maxval=2, dtype=tf.int32), tf.bool)
        image = tf.cond(cond_he,
                        lambda: self._equalize_histogram(image),
                        lambda: tf.identity(image))

        return image, mask

    def mask_generator(self, anno):
        ret, thresh = cv2.threshold(anno.numpy().astype(np.uint8), 127, 255, 0)
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        ellipse = cv2.fitEllipse(contours[0])

        mask = cv2.ellipse(anno.numpy().astype(np.uint8),
                           ellipse, (255, 255, 255), -1)

        return tf.convert_to_tensor(mask, dtype=tf.float32)

    def cut_roi(self, image, anno):
        cond_cut = tf.cast(tf.random.uniform(
            [], maxval=2, dtype=tf.int32), tf.bool)

        mask = tf.cond(cond_cut,
                       lambda: self.mask_generator(anno),
                       lambda: tf.identity(anno))

        image = tf.cond(cond_cut,
                        lambda: (image * mask)/255.,
                        lambda: tf.identity(image))

        return image, mask

    @tf.function
    def map_function(self, images_path, masks_path):
        image, mask = self.parse_data(images_path, masks_path)

        def augmentation_func(image_f, mask_f):
            image_f, mask_f = self.equalize_histogram(image_f, mask_f)
            image_f, mask_f = self.normalize_data(image_f, mask_f)

            if self.augmentation:
                if self.compose:
                    image_f, mask_f = self.change_brightness(image_f, mask_f)
                    image_f, mask_f = self.flip_horizontally(image_f, mask_f)
                    image_f, mask_f = self.rotate(image_f, mask_f)
                    image_f, mask_f = self.shift(image_f, mask_f)
                else:
                    options = [self.change_brightness,
                               self.flip_horizontally,
                               self.rotate,
                               self.shift]
                    augment_func = random.choice(options)
                    image_f, mask_f = augment_func(image_f, mask_f)

            if self.one_hot_encoding:
                if self.palette is None:
                    raise ValueError('No Palette for one-hot encoding specified in the data loader! \
                                      please specify one when initializing the loader.')
                image_f, mask_f = self.one_hot_encode(image_f, mask_f)

            image_f, mask_f = self.resize_data(image_f, mask_f)

            return image_f, mask_f

        return tf.py_function(augmentation_func, [image, mask], [tf.float32, tf.float32])

    @tf.function
    def test_map_function(self, images_path):
        image = self.parse_data(images_path)

        image_f = self.normalize_data(image)
        image_f = self.resize_data(image_f)

        return image_f

    def data_gen(self, batch_size, shuffle=False):
        if self.mode in ["train", "valid"]:
            # Create dataset out of the 2 files:
            data = tf.data.Dataset.from_tensor_slices(
                (self.image_paths, self.mask_paths))

            # Parse images and labels
            data = data.map(self.map_function, num_parallel_calls=AUTOTUNE)
        elif self.mode == "test":
            data = tf.data.Dataset.from_tensor_slices((self.image_paths))
            data = data.map(self.test_map_function,
                            num_parallel_calls=AUTOTUNE)

        if shuffle:
            # Prefetch, shuffle then batch
            data = data.prefetch(AUTOTUNE).shuffle(
                random.randint(0, len(self.image_paths))).batch(batch_size)
        else:
            # Batch and prefetch
            data = data.batch(batch_size).prefetch(AUTOTUNE)

        return data


data = DataLoader("./data/test_set/",
                  mode="test",
                  image_size=config["image_size"])


def read_image_by_tf(path, channels=1):
    image_content = tf.io.read_file(path)
    image = tf.image.decode_png(image_content, channels=channels)

    return tf.cast(image, tf.float32)


def load_infer_image(path, channels=1):
    image = read_image_by_tf(path, channels=channels)
    image = data.normalize_data(image)
    image = data.resize_data(image)

    return image


if __name__ == "__main__":
    data = DataLoader("../data/training_set",
                      augmentation=True,
                      one_hot_encoding=True,
                      palette=[255]).data_gen(32)

    for image, mask in data:
        print(image.shape)
        break
