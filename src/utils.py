import os
import sys
import glob
import datetime
import numpy as np
from PIL import Image

# import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import load_model

import seglosses


def time_to_timestr():
    timestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    return timestr


def load_model_from_path(file_path):
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

    model = load_model(file_path, custom_objects=custom_objects)

    return model


def read_image(path):
    return np.array(Image.open(path))


def batch_2_numpy(data_batches):
    batch_numpy = [batch.numpy().squeeze() for batch in data_batches]
    data = batch_numpy[0]

    for i in range(1, len(batch_numpy)):
        data = np.concatenate((data, batch_numpy[i]), axis=0)

    return data
