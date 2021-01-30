import os
import cv2
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import tensorflow as tf
import matplotlib.pyplot as plt

from reg.data import DataLoader

data = DataLoader("../data/training_set",
                  one_hot_encoding=True,
                  palette=[255])

def plot(image):
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def draw_ellipse(img, paras):
    return cv2.ellipse(img, (paras[0], paras[1]),
                       (paras[2], paras[3]),
                       -paras[4],
                       0,
                       360,
                       color=(0, 255, 255),
                       thickness=3)


def pred_one_model(model, image, image_ori):
    pred = model.predict(tf.expand_dims(image, axis=0))
    pred_image = draw_ellipse(image_ori, pred[0])

    return pred_image

def show_pred(image_path, model):
    image_ori = cv2.imread(image_path)
    image_content = cv2.cvtColor(image_ori, cv2.COLOR_BGR2GRAY)
    image = np.asarray(np.dstack((image_content.squeeze(), image_content.squeeze(), image_content.squeeze())))
    image = tf.convert_to_tensor(image)
    image = tf.cast(image, tf.float32)

    image = data.normalize_data(image)
    image = data.resize_data(image)
    
    pred_image = pred_one_model(model, image, image_ori)
    plot(pred_image)


if __name__ == "__main__":
    show_pred("")