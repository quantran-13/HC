import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import cv2


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


if __name__ == "__main__":
    pass
