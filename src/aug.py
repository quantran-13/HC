import os
import random
import numpy as np
import pandas as pd
from PIL import Image
# import matplotlib.pyplot as plt

import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf

from config import *
from utils import read_image
from ellipse_fitting import *

# https://github.com/HasnainRaz/SemSegPipeline/blob/master/dataloader.py
AUTOTUNE = tf.data.experimental.AUTOTUNE

DF_AUG_TRAIN = []
DF_AUG_VALID = []


class DataLoader(object):
    """
        A TensorFlow Dataset API based loader for semantic segmentation problems.
    """

    def __init__(self, root, mode="train", augmentation=False, compose=False, one_hot_encoding=False, palette=None, image_size=(216, 320, 1), save=False):
        """
        root: "../data/training_set"
        """
        super().__init__()
        self.root = root
        self.mode = mode
        self.augmentation = augmentation
        self.compose = compose
        self.one_hot_encoding = one_hot_encoding
        self.palette = palette
        self.image_size = (image_size[0], image_size[1])
        self.save = save
        if not os.path.exists("../data/aug_{}".format(self.mode)):
            os.mkdir("../data/aug_{}".format(self.mode))

        if (self.mode == "train"):
            self.df = pd.read_csv("../data/train.csv")
        elif (self.mode == "valid"):
            self.df = pd.read_csv("../data/valid.csv")
        elif (self.mode == "test"):
            self.df = pd.read_csv("../data/test_set_pixel_size.csv")

        self.parse_data_path()

    def parse_data_path(self):
        if self.mode in ["train", "valid"]:
            self.image_paths = [os.path.join(self.root, _[0])
                                for _ in self.df.values.tolist()]

            self.mask_paths = [_.replace(".png", "_Annotation.png")
                               for _ in self.image_paths]

            self.factors = [_[1] for _ in self.df.values.tolist()]
        elif self.mode == "test":
            self.image_paths = [os.path.join(self.root, _[0])
                                for _ in self.df.values.tolist()]
            self.factors = [_[1] for _ in self.df.values.tolist()]

    def parse_data(self, image_paths, factors, mask_paths=None):
        image_content = tf.io.read_file(image_paths)
        # grayscale
        images = tf.image.decode_png(image_content, channels=1)
        images = tf.cast(images, tf.float32)

        if self.mode in ["train", "valid"]:
            mask_content = tf.io.read_file(mask_paths)
            # grayscale
            masks = tf.image.decode_png(mask_content, channels=1)
            masks = tf.cast(masks, tf.float32)

            return image_paths, images, factors, masks

        return images, factors

    def cal_ellipse_parameters(self, image_path, factor, mask):
        (xx, yy), (MA, ma), angle = ellipse_fit_anno(mask.numpy().squeeze())
        center_x_mm = factor * yy
        center_y_mm = factor * xx
        semi_axes_a_mm = factor * ma / 2
        semi_axes_b_mm = factor * MA / 2
        angle_rad = (-angle * np.pi / 180) % np.pi
        # print(center_x_mm.numpy(), center_y_mm.numpy(), semi_axes_a_mm.numpy(), semi_axes_b_mm.numpy(), angle_rad)

        if (self.mode == "train"):
            DF_AUG_TRAIN.append([image_path.split("/")[-1],
                                 factor.numpy(),
                                 center_x_mm.numpy(),
                                 center_y_mm.numpy(),
                                 semi_axes_a_mm.numpy(),
                                 semi_axes_b_mm.numpy(),
                                 angle_rad
                                 ])
        elif (self.mode == "valid"):
            DF_AUG_VALID.append([image_path.split("/")[-1],
                                 factor.numpy(),
                                 center_x_mm.numpy(),
                                 center_y_mm.numpy(),
                                 semi_axes_a_mm.numpy(),
                                 semi_axes_b_mm.numpy(),
                                 angle_rad
                                 ])

    def get_image_idx(self, image_path):
        idx = image_path.numpy().decode("utf-8").split("/")[-1].split("_")[0]
        idx = idx + str(random.randint(0, 30000))

        return idx.zfill(8)

    def change_brightness(self, image_path, image, factor, mask):
        """
            Randomly applies a random brightness change.
        """
        cond_brightness = tf.cast(tf.random.uniform(
            [], maxval=2, dtype=tf.int32), tf.bool)
        image = tf.cond(cond_brightness,
                        lambda: tf.image.random_brightness(image, 0.1),
                        lambda: tf.identity(image))

        if self.save:
            idx = self.get_image_idx(image_path)
            image_path = "../data/aug_{}/{}_brightness_HC.png".format(
                self.mode, idx)
            cv2.imwrite(image_path, image.numpy().squeeze())
            cv2.imwrite(image_path.replace(
                ".png", "_Annotation.png"), mask.numpy().squeeze())

        self.cal_ellipse_parameters(image_path, factor, mask)

        return image, mask

    def flip_horizontally(self, image_path, image, factor, mask):
        """
            Randomly flips image and mask horizontally in accord.
        """
        comb_tensor = tf.concat([image, mask], axis=2)
        comb_tensor = tf.image.random_flip_left_right(comb_tensor)
        image, mask = tf.split(comb_tensor, [1, 1], axis=2)

        if self.save:
            idx = self.get_image_idx(image_path)
            image_path = "../data/aug_{}/{}_flip_horizontally_HC.png".format(
                self.mode, idx)
            cv2.imwrite(image_path, image.numpy().squeeze())
            cv2.imwrite(image_path.replace(
                ".png", "_Annotation.png"), mask.numpy().squeeze())

        self.cal_ellipse_parameters(image_path, factor, mask)

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

    def rotate(self, image_path, image, factor, mask):
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

        if self.save:
            idx = self.get_image_idx(image_path)
            image_path = "../data/aug_{}/{}_rotate_HC.png".format(
                self.mode, idx)
            cv2.imwrite(image_path, image.numpy().squeeze())
            cv2.imwrite(image_path.replace(
                ".png", "_Annotation.png"), mask.numpy().squeeze())

        self.cal_ellipse_parameters(image_path, factor, mask)

        return image, mask

    def shift(self, image_path, image, factor, mask):
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

        if self.save:
            idx = self.get_image_idx(image_path)
            image_path = "../data/aug_{}/{}_shift_HC.png".format(
                self.mode, idx)
            cv2.imwrite(image_path, image.numpy().squeeze())
            cv2.imwrite(image_path.replace(
                ".png", "_Annotation.png"), mask.numpy().squeeze())

        self.cal_ellipse_parameters(image_path, factor, mask)

        return image, mask

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

    @tf.function
    def map_function(self, images_path, masks_path, factors):
        image_path, image, factor, mask = self.parse_data(
            images_path, factors, masks_path)

        def augmentation_func(image_path_f, image_f, factor_f, mask_f):
            if self.augmentation:
                if self.compose:
                    image_f, mask_f = self.change_brightness(
                        image_path_f, image_f, factor_f, mask_f)
                    image_f, mask_f = self.flip_horizontally(
                        image_path_f, image_f, factor_f, mask_f)
                    image_f, mask_f = self.rotate(
                        image_path_f, image_f, factor_f, mask_f)
                    image_f, mask_f = self.shift(
                        image_path_f, image_f, factor_f, mask_f)
                else:
                    # print(image_f.shape)
                    options = [self.change_brightness,
                               self.flip_horizontally,
                               self.rotate,
                               self.shift,
                               ]
                    augment_func = random.choice(options)
                    image_f, mask_f = augment_func(
                        image_path_f, image_f, factor_f, mask_f)

            if self.one_hot_encoding:
                if self.palette is None:
                    raise ValueError('No Palette for one-hot encoding specified in the data loader! \
                                      please specify one when initializing the loader.')
                image_f, mask_f = self.one_hot_encode(image_f, mask_f)

            image_f, mask_f = self.resize_data(image_f, mask_f)
            
            return image_f, factor_f, mask_f

        return tf.py_function(augmentation_func, [image_path, image, factor, mask], [tf.float32, tf.float32, tf.float32])

    @tf.function
    def test_map_function(self, images_path):
        image, factor = self.parse_data(images_path)

        image_f = self.normalize_data(image)
        image_f = self.resize_data(image_f)

        return image_f

    def data_gen(self, batch_size, shuffle=False):
        if self.mode in ["train", "valid"]:
            # Create dataset out of the 2 files:
            data = tf.data.Dataset.from_tensor_slices(
                (self.image_paths, self.mask_paths, self.factors))

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


if __name__ == "__main__":
    for i in range(50):
        data = DataLoader("../data/training_set",
                          mode="train",
                          augmentation=True,
                          one_hot_encoding=True,
                          palette=[255],
                          save=True).data_gen(32)

        for image, factor, mask in data:
            pass

    df_aug_train = pd.DataFrame(DF_AUG_TRAIN, columns=[
        "filename", "pixel size(mm)", "center x(mm)", "center y(mm)", "semi axes a(mm)", "semi axes b(mm)", "angle(rad)"])
    df_aug_train.to_csv("../data/aug_train.csv", index=False)

    for i in range(10):
        data = DataLoader("../data/training_set",
                          mode="valid",
                          augmentation=True,
                          one_hot_encoding=True,
                          palette=[255],
                          save=True).data_gen(32)

        for image, factor, mask in data:
            pass

    df_aug_valid = pd.DataFrame(DF_AUG_VALID, columns=[
        "filename", "pixel size(mm)", "center x(mm)", "center y(mm)", "semi axes a(mm)", "semi axes b(mm)", "angle(rad)"])
    df_aug_valid.to_csv("../data/aug_valid.csv", index=False)
