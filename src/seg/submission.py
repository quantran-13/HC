import os
import numpy as np
import pandas as pd

from seg.data import load_infer_image
from seg.predict import pred_one_image
from seg.ellipse_fitting import ellipse_fit_mask
from seg.utils import load_infer_model


def generate_submission(model_path, predicted_path):
    centers_x = list()
    centers_y = list()
    axes_a = list()
    axes_b = list()
    angles = list()

    model = load_infer_model(model_path)

    df = pd.read_csv("./data/test_set_pixel_size.csv")

    for idx, row in df.iterrows():
        image_path = os.path.join("./data/test_set", row["filename"])
        image = load_infer_image(image_path)

        pred_image = pred_one_image(model, image)

        assert 540 / pred_image.shape[0] == 800 / pred_image.shape[1]

        (xx, yy), (MA, ma), angle = ellipse_fit_mask(pred_image)
        factor = row["pixel size(mm)"] * 540 / pred_image.shape[0]

        center_x_mm = factor * yy
        center_y_mm = factor * xx
        semi_axes_a_mm = factor * ma / 2
        semi_axes_b_mm = factor * MA / 2
        angle_rad = (-angle * np.pi / 180) % np.pi

        centers_x.append(center_x_mm)
        centers_y.append(center_y_mm)
        axes_a.append(semi_axes_a_mm)
        axes_b.append(semi_axes_b_mm)
        angles.append(angle_rad)

    df = df.drop(columns="pixel size(mm)")
    df["center_x_mm"] = centers_x
    df["center_y_mm"] = centers_y
    df["semi_axes_a_mm"] = axes_a
    df["semi_axes_b_mm"] = axes_b
    df["angle_rad"] = angles

    print("="*100)
    print("Make submission csv ...")
    df.to_csv(
        "./submission/{}.csv".format(predicted_path.split("/")[-1]), index=False)


if __name__ == "__main__":
    generate_submission()
