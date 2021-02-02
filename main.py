import argparse
import os
import sys
sys.path.append("./src")

import numpy as np
import pandas as pd
from PIL import Image
import cv2

from tensorflow.keras.models import load_model
from seg.utils import load_infer_model
from seg import predict
from reg import infer_reg
from reg.data import DataLoader, read_image_by_tf

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('image_path', type=str, default=None)
    parser.add_argument('--mask_path', type=str, default=None)
    parser.add_argument('--model_path', type=str,
                        default="./models/regression_model.hdf5")
    parser.add_argument('--method', type=str, default='r',
                        help="'r': regression, 's': segmentation")
    return parser.parse_args()

import matplotlib.pyplot as plt
def mm_2_pixel(paras, factor):
    paras_pixel = {}
    paras_pixel["yy"] = paras["center_x_mm"] / factor
    paras_pixel["xx"] = paras["center_y_mm"] / factor
    paras_pixel["ma"] = paras["semi_axes_a_mm"] / factor
    paras_pixel["MA"] = paras["semi_axes_b_mm"] / factor
    paras_pixel = dict(map(lambda x: (x[0], int(x[1])), paras_pixel.items()))
    paras_pixel["angle"] = (paras["angle_rad"] * 180 / np.pi) - 90
    # print(paras_pixel)

    return paras_pixel


def draw_ellipse(image, paras):
    return cv2.ellipse(image,
                      (paras["yy"], paras["xx"]),
                      (paras["MA"], paras["ma"]),
                      paras["angle"],
                      0,
                      360,
                      color=(251, 189, 5),
                      thickness=3)


def show_one_ellipse(image_path, paras, factor):
    _image = np.array(Image.open(image_path))
    image = np.asarray(np.dstack((_image.squeeze(), _image.squeeze(), _image.squeeze())), dtype=np.uint8)
    paras_pixel = mm_2_pixel(paras, factor)
    return image, draw_ellipse(image, paras_pixel)


def show_ellipses(sub_path):
    df = pd.read_csv("../../visualize/pixel_size.csv")
    sub_df = pd.read_csv(sub_path)

    for _, row in sub_df.iterrows():
        factor = df.loc[df["filename"] == row["filename"]]["pixel size(mm)"].values[0]
 
        image_path = "../../visualize/images/{}".format(row["filename"])
        _, image = show_one_ellipse(image_path, row, factor)
        # cv2.imwrite(os.path.join("./outputs/A3", row["filename"]), image)
        # print(row["filename"])

        img_anno = row['filename'].rstrip('.png') + "_Annotation.png"
        mask_path = os.path.join("../../visualize/images", img_anno)

        mask = read_image_by_tf(mask_path)
        mask = np.asarray(np.dstack((mask.numpy()/255 * 64, mask.numpy()/255 * 134, mask.numpy()/255 * 244)), dtype=np.uint8)
        image = cv2.addWeighted(image, 0.7, mask, 1, 0)
        Image.fromarray(image.astype(np.uint8)).save(os.path.join('./outputs/A14', row["filename"]))
    


if __name__ == "__main__":
    # args = parse_args()
    # image_path = args.image_path
    # mask_path = args.mask_path
    # model_path = args.model_path
    # image_list = ["009_HC", "064_2HC", "227_HC", "647_HC"]

    # model_list = os.listdir("./models")

    # pred_file = pd.read_csv("../../visualize/file/A3.csv")


    # if args.method == 'r':
    show_ellipses("../../visualize/file/A14.csv")

    # if args.method == 's':
    #     model = load_infer_model(model_path)
    #     predict.plot_pred(model, image_path, mask_path)
