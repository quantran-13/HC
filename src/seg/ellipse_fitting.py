import numpy as np
import pandas as pd

import cv2
import matplotlib.pyplot as plt

import tensorflow as tf

from seg.config import config
from seg.predict import pred_one_image
from seg.utils import load_infer_model
from seg.data import load_infer_image, read_image_by_tf


def ellipse_fit(points, method="Direct"):
    if method == "AMS":
        (xx, yy), (MA, ma), angle = cv2.fitEllipseAMS(points)
    elif method == "Direct":
        (xx, yy), (MA, ma), angle = cv2.fitEllipseDirect(points)
    elif method == "Simple":
        (xx, yy), (MA, ma), angle = cv2.fitEllipse(points)

    return (xx, yy), (MA, ma), angle


def ellipse_fit_mask(binary_mask, method="Direct"):
    assert binary_mask.min() >= 0.0 and binary_mask.max() <= 1.0
    points = np.argwhere(binary_mask > 0.5)  # TODO: tune threshold
    (xx, yy), (MA, ma), angle = ellipse_fit(points)

    return (xx, yy), (MA, ma), angle


def ellipse_fit_anno(anno, method="Direct"):
    points = np.argwhere(anno > 127)
    (xx, yy), (MA, ma), angle = ellipse_fit(points)

    return (xx, yy), (MA, ma), angle


def draw_ellipse(img, binary_mask):
    (xx, yy), (MA, ma), angle = ellipse_fit_mask(binary_mask)
    img = cv2.ellipse(img,
                      (int(yy), int(xx)),
                      (int(ma / 2), int(MA / 2)),
                      -angle,
                      0,
                      360,
                      color=(251, 189, 5),
                      thickness=1)

    return img


def ellipse_circumference_approx(major_semi_axis, minor_semi_axis):
    h = (major_semi_axis - minor_semi_axis) ** 2 / \
        (major_semi_axis + minor_semi_axis) ** 2
    circ = (np.pi * (major_semi_axis + minor_semi_axis)
            * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h))))

    return circ


def rotate_point(point, center, deg):
    point = np.asarray(point)
    center = np.asarray(center)
    point = point - center

    rad = math.radians(deg)
    rotMatrix = np.array([[math.cos(rad), math.sin(rad)],
                          [-math.sin(rad), math.cos(rad)]])
    rotated = np.dot(rotMatrix, point).astype(np.int)

    return tuple(rotated + center)


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
    pass
