import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from seg.ellipse_fitting import ellipse_fit_anno, ellipse_circumference_approx, rotate_point


def read_image(path):
    return np.array(Image.open(path))


def generate_train_valid_indices():
    X = np.arange(999)
    X_train, X_valid, _, _ = train_test_split(
        X, np.ones_like(X), test_size=0.2)

    train_indices = X_train
    valid_indices = X_valid

    assert len(train_indices) + len(valid_indices) == 999
    assert not set(train_indices) & set(valid_indices)

    np.save("./data/train_indices.npy", train_indices)
    np.save("./data/valid_indices.npy", valid_indices)


def add_ellipses_csv():
    """
        Add ellipses parameters & bounding box parameters to training_set_pixel_size_and_HC.csv
    """

    df = pd.read_csv("./data/training_set_pixel_size_and_HC.csv")

    df_pixel = df.copy()
    centers_x_pixel = list()
    centers_y_pixel = list()
    axes_a_pixel = list()
    axes_b_pixel = list()
    angles_pixel = list()

    df_mm = df.copy()
    centers_x_mm = list()
    centers_y_mm = list()
    axes_a_mm = list()
    axes_b_mm = list()
    angles_mm = list()

    for i, row in df.iterrows():
        filename = row["filename"]
        print("Image: {}".format(filename))
        anno = read_image(os.path.join("./data/training_set",
                                       filename.replace(".png", "_Annotation.png")))

        # plt.imshow(anno)
        (xx, yy), (MA, ma), angle = ellipse_fit_anno(anno)

        center_x_pixel = yy
        center_y_pixel = xx
        semi_axes_a_pixel = ma / 2
        semi_axes_b_pixel = MA / 2
        angle_rad = (-angle * np.pi / 180) % np.pi

        centers_x_pixel.append(center_x_pixel)
        centers_y_pixel.append(center_y_pixel)
        axes_a_pixel.append(semi_axes_a_pixel)
        axes_b_pixel.append(semi_axes_b_pixel)
        angles_pixel.append(angle_rad)
        # print("In pixel:", center_x_pixel, center_y_pixel,
        #       semi_axes_a_pixel, semi_axes_b_pixel, angle_rad)

        factor = row["pixel size(mm)"]
        center_x_mm = factor * yy
        center_y_mm = factor * xx
        semi_axes_a_mm = factor * ma / 2
        semi_axes_b_mm = factor * MA / 2
        angle_rad = (-angle * np.pi / 180) % np.pi

        centers_x_mm.append(center_x_mm)
        centers_y_mm.append(center_y_mm)
        axes_a_mm.append(semi_axes_a_mm)
        axes_b_mm.append(semi_axes_b_mm)
        angles_mm.append(angle_rad)
        # print("In mm:", center_x_mm, center_y_mm,
        #       semi_axes_a_mm, semi_axes_b_mm, angle_rad)

        circ = ellipse_circumference_approx(semi_axes_a_mm, semi_axes_b_mm)
        assert np.abs(circ - row["head circumference (mm)"]
                      ) < 0.1, "Wrong ellipse circumference approximation"
        # print("circ: ", circ)
        # print("true circ: ", row["head circumference (mm)"])

    df_pixel["center_x_pixel"] = centers_x_pixel
    df_pixel["center_y_pixel"] = centers_y_pixel
    df_pixel["semi_axes_a_pixel"] = axes_a_pixel
    df_pixel["semi_axes_b_pixel"] = axes_b_pixel
    df_pixel["angle_rad"] = angles_mm

    df_mm["center_x_mm"] = centers_x_mm
    df_mm["center_y_mm"] = centers_y_mm
    df_mm["semi_axes_a_mm"] = axes_a_mm
    df_mm["semi_axes_b_mm"] = axes_b_mm
    df_mm["angle_rad"] = angles_mm

    df_pixel.to_csv(
        "./data/training_set_pixel_size_and_HC_and_ellipses_in_pixel.csv", index=False)

    df_mm.to_csv(
        "./data/training_set_pixel_size_and_HC_and_ellipses.csv", index=False)


def add_ellipses_keypoints():
    df = pd.read_csv("./data/training_set_pixel_size_and_HC.csv")

    df_kp = df.copy()
    x0s = list()
    y0s = list()
    x1s = list()
    y1s = list()
    x2s = list()
    y2s = list()
    x3s = list()
    y3s = list()
    x4s = list()
    y4s = list()

    for i, row in df.iterrows():
        filename = row["filename"]
        print("Image: {}".format(filename))
        anno = read_image(os.path.join("./data/training_set",
                                       filename.replace(".png", "_Annotation.png")))

        # plt.imshow(anno)
        (xx, yy), (MA, ma), angle = ellipse_fit_anno(anno)

        center = (yy, xx)
        x0s.append(yy)
        y0s.append(xx)

        point = (yy + ma/2, xx)
        point_rotated = rotate_point(point, center, angle)
        x1s.append(point_rotated[0])
        y1s.append(point_rotated[1])

        point = (yy, xx + MA/2)
        point_rotated = rotate_point(point, center, angle)
        x2s.append(point_rotated[0])
        y2s.append(point_rotated[1])

        point = (yy - ma/2, xx)
        point_rotated = rotate_point(point, center, angle)
        x3s.append(point_rotated[0])
        y3s.append(point_rotated[1])

        point = (yy, xx - MA/2)
        point_rotated = rotate_point(point, center, angle)
        x4s.append(point_rotated[0])
        y4s.append(point_rotated[1])

    df_kp["x0"] = x0s
    df_kp["y0"] = y0s
    df_kp["x1"] = x1s
    df_kp["y1"] = y1s
    df_kp["x2"] = x2s
    df_kp["y2"] = y2s
    df_kp["x3"] = x3s
    df_kp["y3"] = y3s
    df_kp["x4"] = x4s
    df_kp["y4"] = y4s

    df_kp.to_csv(
        "./data/training_set_pixel_size_and_HC_and_ellipses_keypoints.csv", index=False)


def create_data_csv(df, npy_file, out_file):
    npy_path = "./data/{}".format(npy_file)

    subset_indices = np.load(npy_path)

    subset_df = df[df.index.isin(subset_indices)]
    subset_df.to_csv("./data/{}".format(out_file), index=False)


def generate_data_csv(data_csv_path, train_file, valid_file):
    df = pd.read_csv("./data/{}".format(data_csv_path))

    create_data_csv(df, "train_indices.npy", train_file)
    create_data_csv(df, "valid_indices.npy", valid_file)


if __name__ == "__main__":
    L1 = os.listdir("../data")

    L2 = ["training_set_pixel_size_and_HC_and_ellipses.csv",
          "training_set_pixel_size_and_HC_and_ellipses_in_pixel.csv",
          "training_set_pixel_size_and_HC_and_ellipses_keypoints.csv"]
    if any(x not in L1 for x in L2):
        add_ellipses_csv()
        add_ellipses_keypoints()

    if "train_indices.npy" not in os.listdir("../data"):
        generate_train_valid_indices()

    if "train.csv" not in os.listdir("../data"):
        generate_data_csv(
            "training_set_pixel_size_and_HC_and_ellipses.csv", "train.csv", "valid.csv")
        generate_data_csv(
            "training_set_pixel_size_and_HC_and_ellipses_in_pixel.csv", "train_in_pixel.csv", "valid_in_pixel.csv")
        generate_data_csv(
            "training_set_pixel_size_and_HC_and_ellipses_keypoints.csv", "train_keypoints.csv", "valid_keypoints.csv")
