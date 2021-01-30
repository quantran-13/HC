import numpy as np
import pandas as pd

import cv2
import matplotlib.pyplot as plt

from seg.config import config
from seg.data import DataLoader
from seg.predict import pred_one_image
from seg.utils import read_image_by_tf


data = DataLoader("../../data/test_set/",
                  mode="test",
                  image_size=config["image_size"])


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
                      color=(255, 255, 255),
                      thickness=2)

    return img


def ellipse_circumference_approx(major_semi_axis, minor_semi_axis):
    h = (major_semi_axis - minor_semi_axis) ** 2 / \
        (major_semi_axis + minor_semi_axis) ** 2
    circ = (np.pi * (major_semi_axis + minor_semi_axis)
            * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h))))

    return circ


def plot(image):
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis('off')


def plot_pred(model, image_path):
    image = read_image_by_tf(image_path)
    image = data.normalize_data(image)
    image = data.resize_data(image)

    pred_image = pred_one_image(model, image)

    # plot(pred_image)
    cv2.imshow("img", pred_image)
    cv2.waitKey(0)


if __name__ == "__main__":
    pass
