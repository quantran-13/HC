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
