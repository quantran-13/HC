import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import tensorflow as tf
from config import *


AUTOTUNE = tf.data.experimental.AUTOTUNE


def generate_train_valid_indices():
    output_dir = "../data/"

    X = np.arange(999)
    X_train, X_valid, _, _ = train_test_split(
        X, np.ones_like(X), test_size=0.2)

    train_indices = X_train
    valid_indices = X_valid

    assert len(train_indices) + len(valid_indices) == 999
    assert not set(train_indices) & set(valid_indices)

    np.save(os.path.join(output_dir, "train_indices.npy"), train_indices)
    np.save(os.path.join(output_dir, "valid_indices.npy"), valid_indices)


def create_data_csv(df, mode):
    output_dir = "../data/"
    output_path = os.path.join(output_dir, "{}.csv".format(mode))
    npy_path = "../data/{}_indices.npy".format(mode)

    subset_indices = np.load(npy_path)

    subset_df = df[df.index.isin(subset_indices)]
    subset_df = subset_df["filename"]
    subset_df.to_csv(output_path, index=False)


def generate_data_csv():
    df = pd.read_csv("../data/training_set_pixel_size_and_HC.csv")

    create_data_csv(df, "train")
    create_data_csv(df, "valid")


class DataLoader(object):
    """
    A TensorFlow Dataset API based loader for semantic segmentation problems.
    """

    def __init__(self, root, mode="train", one_hot_encoding=True, palette=PALETTE, image_size=(216, 320, 1)):
        """
        root: "../data/training_set"
        """
        super().__init__()
        self.root = root
        self.one_hot_encoding = one_hot_encoding
        self.palette = palette
        self.image_size = (image_size[0], image_size[1])

        if (mode == "train"):
            self.df = pd.read_csv("../data/train.csv")
        elif (mode == "valid"):
            self.df = pd.read_csv("../data/valid.csv")
        elif (mode == "test"):
            self.df = pd.read_csv("../data/test_set_pixel_size.csv")

        self.parse_data_path()

    def parse_data_path(self):
        self.image_paths = [os.path.join(self.root, _[0])
                            for _ in self.df.values.tolist()]
        
        self.mask_paths = [_.replace(".png", "_Annotation.png")
                           for _ in self.image_paths]

    def parse_data(self, image_paths, mask_paths):
        image_content = tf.io.read_file(image_paths)
        mask_content = tf.io.read_file(mask_paths)

        images = tf.image.decode_png(image_content, channels=1)
        images = tf.cast(images, tf.float32)
        
        masks = tf.image.decode_png(mask_content, channels=1)
        masks = tf.cast(masks, tf.float32)

        return images, masks

    def resize_data(self, image, mask):
        """
        Resizes images to specified size.
        """
        image = tf.image.resize(image, self.image_size, method="nearest")
        mask = tf.image.resize(mask, self.image_size, method="nearest")

        return image, mask

    def one_hot_encode(self, image, mask):
        one_hot_map = []

        for colour in self.palette:
            class_map = tf.reduce_all(tf.equal(mask, colour), axis=-1)
            one_hot_map.append(class_map)

        one_hot_map = tf.stack(one_hot_map, axis=-1)
        one_hot_map = tf.cast(one_hot_map, tf.float32)

        return image, one_hot_map

    @tf.function
    def map_function(self, images_path, masks_path):
        image, mask = self.parse_data(images_path, masks_path)

        def augmentation_func(image_f, mask_f):
            if self.one_hot_encoding:
                if self.palette is None:
                    raise ValueError('No Palette for one-hot encoding specified in the data loader! \
                                      please specify one when initializing the loader.')
                image_f, mask_f = self.one_hot_encode(image_f, mask_f)

            image_f, mask_f = self.resize_data(image_f, mask_f)

            return image_f, mask_f

        return tf.py_function(augmentation_func, [image, mask], [tf.float32, tf.float32])

    def data_gen(self, batch_size, shuffle=False):
        # Create dataset out of the 2 files:
        data = tf.data.Dataset.from_tensor_slices(
            (self.image_paths, self.mask_paths))

        # Parse images and labels
        data = data.map(self.map_function, num_parallel_calls=AUTOTUNE)

        if shuffle:
            # Prefetch, shuffle then batch
            data = data.prefetch(AUTOTUNE).shuffle(
                random.randint(0, len(self.image_paths))).batch(batch_size)
        else:
            # Batch and prefetch
            data = data.batch(batch_size).prefetch(AUTOTUNE)

        return data


if __name__ == "__main__":
    data = DataLoader("../data/training_set").data_gen(32)
    for image, mask in data:
        print(np.unique(image.numpy()))
        break
