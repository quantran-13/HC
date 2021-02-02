import os
import cv2
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import tensorflow as tf
import matplotlib.pyplot as plt

from reg.data import DataLoader, read_image_by_tf

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
                       color=(251, 189, 5),
                       thickness=2)


def pred_one_model(model, image, image_ori):
    pred = model.predict(tf.expand_dims(image, axis=0))
    pred_image = draw_ellipse(image_ori, pred[0])

    return pred_image

def show_pred(image_path, model,  img_id, mask_path=None):
    image_ori = cv2.imread(image_path)
    h, w,_ = image_ori.shape
    image_content = cv2.cvtColor(image_ori, cv2.COLOR_BGR2GRAY)
    image = np.asarray(np.dstack((image_content.squeeze(), image_content.squeeze(), image_content.squeeze())))
    image = tf.convert_to_tensor(image)
    image = tf.cast(image, tf.float32)

    if mask_path:
        mask = read_image_by_tf(mask_path)
        mask = np.asarray(np.dstack((mask.numpy()/255 * 64, mask.numpy()/255 * 134, mask.numpy()/255 * 244)), dtype=np.uint8)
        image_ori = cv2.addWeighted(image_ori, 0.7, mask, 1, 0)

    image = data.normalize_data(image)
    image = data.resize_data(image)
    
    pred_image = pred_one_model(model, image, image_ori)
    Image.fromarray(pred_image.astype(np.uint8)).save(os.path.join('./outputs/A14', img_id+".png"))
    # plot(pred_image)


if __name__ == "__main__":
    show_pred("")