import os
import numpy as np

import math
import datetime
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model

from seg import seglosses


def time_to_timestr():
    timestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    return timestr


def load_pretrain_model(file_path):
    custom_objects = {
        "jaccard_loss": seglosses.jaccard_loss,
        "jaccard_index": seglosses.jaccard_index,
        "dice_loss": seglosses.dice_loss,
        "dice_coeff": seglosses.dice_coeff,
        "bce_loss": seglosses.bce_loss,
        "bce_dice_loss": seglosses.bce_dice_loss,
        "loss": seglosses.focal_loss(),
        "f_d_loss": seglosses.focal_dice_loss()
    }

    return load_model(file_path, custom_objects=custom_objects)


def load_infer_model(file_path):
    return load_model(file_path, compile=False)


def read_image(path):
    return np.array(Image.open(path))


def read_image_by_tf(path):
    image_content = tf.io.read_file(path)
    image = tf.image.decode_png(image_content, channels=1)

    return tf.cast(image, tf.float32)


def rotate_point(point, center, deg):
    point = np.asarray(point)
    center = np.asarray(center)
    point = point - center

    rad = math.radians(deg)
    rotMatrix = np.array([[math.cos(rad), math.sin(rad)],
                          [-math.sin(rad), math.cos(rad)]])
    rotated = np.dot(rotMatrix, point).astype(np.int)

    return tuple(rotated + center)
