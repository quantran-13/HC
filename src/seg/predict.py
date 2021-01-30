import os
import numpy as np
import pandas as pd

import cv2
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

import tensorflow as tf

from seg.config import config
from seg.ellipse import draw_ellipse
from seg.utils import load_infer_model
from seg.data import DataLoader, read_image_by_tf, load_infer_image


def eval(model_path):
    # load model
    model = load_infer_model(model_path)

    # load eval data
    print("="*100)
    print("Model trained using 799 images and validated with 200 images. Evaluates using valid set ...")
    valid_set = DataLoader("./data/training_set/",
                           mode="valid",
                           augmentation=True,
                           one_hot_encoding=True,
                           palette=config["palette"],
                           image_size=config["image_size"])
    valid_gen = valid_set.data_gen(config["batch_size"], shuffle=True)

    result = model.evaluate(valid_gen, verbose=0)
    print("Model'score: {} \nLoss: {}.".format(result[1], result[0]))


def pred_one_image(model, image):
    pred_image = model.predict(tf.expand_dims(image, axis=0))

    return pred_image.squeeze()


def pred(model_path, save_path="./data/predcited"):
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # load model
    model = load_infer_model(model_path)

    # load test data
    print("="*100)
    print("Loading testing data ...\n")
    df = pd.read_csv("./data/test_set_pixel_size.csv")

    print("="*100)
    print("Predicting...")
    for idx, row in df.iterrows():
        image_path = os.path.join("./data/test_set", row["filename"])
        image = load_infer_image(image_path)

        pred_image = pred_one_image(model, image)
        pre_paths = image_path.replace(".png", "_Predicted_Mask.png")

        pred_image = Image.fromarray(pred_image * 255).convert("L")
        pred_image.save(pre_paths)


def plot(image):
    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def plot_pred(model, image_path, mask_path=None):

    image_ori = read_image_by_tf(image_path)
    image_ori = tf.image.resize_with_pad(
        image_ori, config["image_size"][0], config["image_size"][1], method="nearest")
    image_ori = np.asarray(np.dstack(
        (image_ori.numpy(), image_ori.numpy(), image_ori.numpy())), dtype=np.uint8)
    plot(image_ori)

    if mask_path:
        mask = read_image_by_tf(mask_path)
        mask = tf.image.resize_with_pad(
            mask, config["image_size"][0], config["image_size"][1], method="nearest")
        mask = np.asarray(np.dstack((mask.numpy(
        )/255 * 64, mask.numpy()/255 * 134, mask.numpy()/255 * 244)), dtype=np.uint8)
        plot(mask)

    image = load_infer_image(image_path)
    pred_mask = pred_one_image(model, image)
    pred_image = np.asarray(
        np.dstack((pred_mask * 234, pred_mask * 68, pred_mask * 53)), dtype=np.uint8)
    plot(pred_image)

    plot_image = cv2.addWeighted(image_ori, 0.7, mask, 1, 0)
    plot_image = cv2.addWeighted(plot_image, 0.7, pred_image, 1, 0)
    plot_image = draw_ellipse(plot_image, pred_mask)
    plot(plot_image)


if __name__ == "__main__":
    pred()
