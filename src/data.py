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


def generate_train_valid_indices():
    X = np.arange(999)
    X_train, X_valid, _, _ = train_test_split(
        X, np.ones_like(X), test_size=0.2)

    train_indices = X_train
    valid_indices = X_valid

    assert len(train_indices) + len(valid_indices) == 999
    assert not set(train_indices) & set(valid_indices)

    np.save("../data/train_indices.npy", train_indices)
    np.save("../data/valid_indices.npy", valid_indices)


def add_ellipses_and_bboxs_csv():
    """
        Add ellipses parameters & bounding box parameters to training_set_pixel_size_and_HC.csv
    """

    df = pd.read_csv("../data/training_set_pixel_size_and_HC.csv")

    centers_x = list()
    centers_y = list()
    axes_a = list()
    axes_b = list()
    angles = list()

    x_mins = list()
    y_mins = list()
    x_maxs = list()
    y_maxs = list()

    for i, row in df.iterrows():
        filename = row["filename"]
        print("Image: {}\n".format(filename))
        anno = read_image(os.path.join("../data/training_set",
                                       filename.replace(".png", "_Annotation.png")))

        # plt.imshow(anno)
        (xx, yy), (MA, ma), angle = ellipse_fit_anno(anno)
        factor = row["pixel size(mm)"]

        center_x_mm = factor * yy
        center_y_mm = factor * xx
        semi_axes_a_mm = factor * ma / 2
        semi_axes_b_mm = factor * MA / 2
        angle_rad = (-angle * np.pi / 180) % np.pi
        # print(center_x_mm, center_y_mm, semi_axes_a_mm, semi_axes_b_mm, angle_rad)

        centers_x.append(center_x_mm)
        centers_y.append(center_y_mm)
        axes_a.append(semi_axes_a_mm)
        axes_b.append(semi_axes_b_mm)
        angles.append(angle_rad)

        circ = ellipse_circumference_approx(semi_axes_a_mm, semi_axes_b_mm)

        assert np.abs(circ - row["head circumference (mm)"]
                      ) < 0.1, "Wrong ellipse circumference approximation"

        # print("circ: ", circ)
        # print("true circ: ", row["head circumference (mm)"])

        points = np.argwhere(anno > 127)
        x_max, y_max = np.amax(points, axis=0)
        x_min, y_min = np.amin(points, axis=0)
        # print(y_min, x_min, y_max, x_max)

        # img = cv2.rectangle(img, (y_min, x_min), (y_max, x_max), (255, 0, 0), 1)
        # plt.imshow(img)

        x_mins.append(y_min)
        y_mins.append(x_min)
        x_maxs.append(y_max)
        y_maxs.append(x_max)

    df["center x(mm)"] = centers_x
    df["center y(mm)"] = centers_y
    df["semi axes a(mm)"] = axes_a
    df["semi axes b(mm)"] = axes_b
    df["angle(rad)"] = angles

    df["x min"] = x_mins
    df["y min"] = y_mins
    df["x max"] = x_maxs
    df["y max"] = y_maxs

    print(df)
    df.to_csv(
        "../data/training_set_pixel_size_and_HC_and_ellipses_and_bounding_boxs.csv", index=False)


def create_data_csv(df, mode):
    npy_path = "../data/{}_indices.npy".format(mode)

    subset_indices = np.load(npy_path)

    subset_df = df[df.index.isin(subset_indices)]
    subset_df.to_csv("../data/{}.csv".format(mode), index=False)


def generate_data_csv():
    df = pd.read_csv(
        "../data/training_set_pixel_size_and_HC_and_ellipses_and_bounding_boxs.csv")

    create_data_csv(df, "train")
    create_data_csv(df, "valid")


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

        if self.save:
            idx = random.randint(0, 30000)
            cv2.imwrite("../data/aug/brightness_{}.png".format(idx),
                        image.numpy().squeeze() * 255)
            cv2.imwrite("../data/aug/brightness_{}_mask.png".format(idx),
                        mask.numpy().squeeze() * 255)

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
        if self.save:
            idx = random.randint(0, 30000)
            cv2.imwrite("../data/aug/contrast_{}.png".format(idx),
                        image.numpy().squeeze() * 255)
            cv2.imwrite("../data/aug/contrast_{}_mask.png".format(idx),
                        mask.numpy().squeeze() * 255)

        return image, mask

    def flip_horizontally(self, image, mask):
        """
            Randomly flips image and mask horizontally in accord.
        """
        comb_tensor = tf.concat([image, mask], axis=2)
        comb_tensor = tf.image.random_flip_left_right(comb_tensor)
        image, mask = tf.split(comb_tensor, [1, 1], axis=2)

        if self.save:
            idx = random.randint(0, 30000)
            cv2.imwrite("../data/aug/flip_horizontally_{}.png".format(idx),
                        image.numpy().squeeze() * 255)
            cv2.imwrite("../data/aug/flip_horizontally_{}_mask.png".format(idx),
                        mask.numpy().squeeze() * 255)

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

        if self.save:
            idx = random.randint(0, 30000)
            cv2.imwrite("../data/aug/rotate_{}.png".format(idx),
                        image.numpy().squeeze() * 255)
            cv2.imwrite("../data/aug/rotate_{}_mask.png".format(idx),
                        mask.numpy().squeeze() * 255)

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

        if self.save:
            idx = random.randint(0, 30000)
            cv2.imwrite("../data/aug/shift_{}.png".format(idx),
                        image.numpy().squeeze() * 255)
            cv2.imwrite("../data/aug/shift_{}_mask.png".format(idx),
                        mask.numpy().squeeze() * 255)

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

        if self.save:
            idx = random.randint(0, 30000)
            cv2.imwrite("../data/aug/zoom_{}.png".format(idx),
                        image.numpy().squeeze() * 255)
            cv2.imwrite("../data/aug/zoom_{}_mask.png".format(idx),
                        mask.numpy().squeeze() * 255)

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
                        lambda: tf.image.random_contrast(image, 0.1, 0.5),
                        lambda: tf.identity(image))

        if self.save:
            idx = random.randint(0, 30000)
            cv2.imwrite("../data/aug/clahe_{}.png".format(idx),
                        image.numpy().squeeze())
            cv2.imwrite("../data/aug/clahe_{}_mask.png".format(idx),
                        mask.numpy().squeeze())

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
                        lambda: (image*mask)/255.,
                        lambda: tf.identity(image))

        if self.save:
            idx = random.randint(0, 30000)
            cv2.imwrite("../data/aug/cut_roi_{}.png".format(idx),
                        image.numpy().squeeze() * 255)
            cv2.imwrite("../data/aug/cut_roi_{}_mask.png".format(idx),
                        mask.numpy().squeeze() * 255)

        return image, mask

    @tf.function
    def map_function(self, images_path, masks_path):
        image, mask = self.parse_data(images_path, masks_path)

        def augmentation_func(image_f, mask_f):
            image_f, mask_f = self.equalize_histogram(image_f, mask_f)
            # image_f, mask_f = self.normalize_data(image_f, mask_f)

            if self.augmentation:
                if self.compose:
                    # images_f, mask_f = self.cut_roi(image_f, mask_f)
                    image_f, mask_f = self.change_brightness(image_f, mask_f)
                    image_f, mask_f = self.flip_horizontally(image_f, mask_f)
                    image_f, mask_f = self.rotate(image_f, mask_f)
                    image_f, mask_f = self.shift(image_f, mask_f)
                    image_f, mask_f = self.zoom(image_f, mask_f)
                else:
                    options = [self.change_brightness,
                               self.flip_horizontally,
                               self.rotate,
                               self.shift,
                               self.zoom]
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


if __name__ == "__main__":

    if "training_set_pixel_size_and_HC_and_ellipses_and_bounding_boxs.csv" not in os.listdir("../data"):
        add_ellipses_and_bboxs_csv()

    if "train_indices.npy" not in os.listdir("../data"):
        generate_train_valid_indices()
        generate_data_csv()

    data = DataLoader("../data/training_set",
                      one_hot_encoding=True,
                      palette=[255],
                      save=True).data_gen(32)

    for image, mask in data:
        pass
