import os
import cv2
import glob
import random

import numpy as np
import pandas as pd

import tensorflow as tf 
from PIL import ImageFile, Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array

import matplotlib.pyplot as plt

AUTOTUNE = tf.data.experimental.AUTOTUNE


class DataLoader(object):
    def __init__(self, root="", mode="train", augmentation=False, compose=False, one_hot_encoding=False, palette=None, image_size=(224, 224, 1), normalize_label=False):
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
        self.channels = image_size[2]

        self.normalize_label = normalize_label
        if self.normalize_label:
            self.get_labels_min_max()

    def get_labels_min_max(self):
        df = pd.read_csv("./data/train_in_pixel.csv")
        list_features = ["center x(mm)", "center y(mm)", "semi axes a(mm)", "semi axes b(mm)", "angle(rad)"]
        des = df[list_features].describe().T

        self.labels_min_max = des[["min", "max"]]

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

        images = tf.image.decode_png(image_content, channels=self.channels)
        images = tf.cast(images, tf.float32)

        if self.mode in ["train", "valid"]:
            mask_content = tf.io.read_file(mask_paths)

            masks = tf.image.decode_png(mask_content, channels=self.channels)
            masks = tf.cast(masks, tf.float32)

            return image_paths, images, masks

        return images

    def normalize_data(self, image, mask=None):
        """
            Normalizes image
        """
        image /= 255.

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

    def flip_horizontally(self, image, mask):
        """
            Randomly flips image and mask horizontally in accord.
        """
        comb_tensor = tf.concat([image, mask], axis=2)
        comb_tensor = tf.image.random_flip_left_right(comb_tensor)
        image, mask = tf.split(comb_tensor, [3, 3], axis=2)

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
        image, mask = tf.split(comb_tensor, [3, 3], axis=2)

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
        image, mask = tf.split(comb_tensor, [3, 3], axis=2)

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
        image, mask = tf.split(comb_tensor, [3, 3], axis=2)

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

    def normalize_labels(self, labels):
        for _, row in enumerate(self.labels_min_max.iterrows()):
            labels[_] = (labels[_] - row[1]["min"]) / (row[1]["max"] - row[1]["min"])

        return (labels[0], labels[1]), (labels[2], labels[3]), labels[4]

    @tf.function
    def map_function(self, images_path, masks_path):
        image_path, image, mask = self.parse_data(images_path, masks_path)

        def augmentation_func(image_path_f, image_f, mask_f):
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
                    raise ValueError("No Palette for one-hot encoding specified in the data loader! \
                                      please specify one when initializing the loader.")
                image_f, mask_f = self.one_hot_encode(image_f, mask_f)

            (center_x_mm, center_y_mm), (semi_axes_a_mm, semi_axes_b_mm), angle_rad = self.mask_to_ellipse_parameters(image_path_f, mask_f)
            image_f, mask_f = self.resize_data(image_f, mask_f)

            return image_f, [center_x_mm, center_y_mm, semi_axes_a_mm, semi_axes_b_mm, angle_rad]

        return tf.py_function(augmentation_func, [image_path, image, mask], [tf.float32, tf.float32])

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