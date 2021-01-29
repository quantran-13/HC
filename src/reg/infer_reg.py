import os
import cv2
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from reg.data import DataLoader

data = DataLoader("../data/training_set",
                  one_hot_encoding=True,
                  palette=[255])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('IMAGE_PATH', type=str)
    return parser.parse_args()

def draw_ellipse(img, paras):
    return cv2.ellipse(img, (paras[0], paras[1]), 
                        (paras[2], paras[3]), 
                        -paras[4], 
                        0, 
                        360, 
                        color=(0, 255, 255), 
                        thickness=3)

def plot(image):
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap="gray")
    plt.axis('off')

def pred_one_model(model_path, image, image_ori):
    model = load_model(model_path, compile=False)
    pred = model.predict(tf.expand_dims(image, axis=0))
    pred_image = draw_ellipse(image_ori, pred[0])

    return pred_image

def show_pred(image_paths):
    image_content = cv2.imread(image_paths, cv2.IMREAD_GRAYSCALE)
    # image_content = tf.io.read_file(image_paths)
    # image = tf.image.decode_png(image_content, channels=3)
    image = np.asarray(np.dstack((image_content.squeeze(), image_content.squeeze(), image_content.squeeze())))
    image = tf.convert_to_tensor(image)
    image = tf.cast(image, tf.float32)
    # image_ori = image.numpy()
    # image_ori = np.asarray(image_ori, dtype=np.uint8)
    image_ori = cv2.imread(image_paths)
    # plot(image_ori)

    # preprocess data
    image = data.normalize_data(image)
    image = data.resize_data(image)
    
    model_path = "./models/sequential_mse=1242.16_Adam_ep290.hdf5"
    pred_image = pred_one_model(model_path, image, image_ori)
    # plot(pred_image)
    cv2.imshow("img", pred_image)
    cv2.waitKey(0)

if __name__ == "__main__":
    args = parse_args()
    show_pred(args.IMAGE_PATH)