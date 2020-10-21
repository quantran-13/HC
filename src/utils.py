import os
import sys
import glob
import datetime
import numpy as np
from PIL import Image

# import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import load_model

import losses


def time_to_timestr():
    timestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    return timestr


def load_model_from_path(file_path):
    custom_objects = {"jaccard_loss": losses.jaccard_loss,
                      "jaccard_index": losses.jaccard_index,
                      "dice_loss": losses.dice_loss,
                      "dice_coeff": losses.dice_coeff}

    model = load_model(file_path, custom_objects=custom_objects)

    return model


def batch_2_numpy(data_batches):
    batch_numpy = [batch.numpy().squeeze() for batch in data_batches]
    data = batch_numpy[0]

    for i in range(1, len(batch_numpy)):
        data = np.concatenate((data, batch_numpy[i]), axis=0)

    return data


def read_predict_2_numpy(predicted_path):
    pre_paths = glob.glob(predicted_path + "/*.png")
    pre_images = np.array(Image.open(pre_paths[0]))/255.

    for i in range(1, len(pre_paths)):
        image = np.array(Image.open(pre_paths[i]))/255.
        pre_images = np.dstack((pre_images, image))

    return np.moveaxis(pre_images, -1, 0)
