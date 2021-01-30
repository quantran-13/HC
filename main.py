import argparse

import sys
sys.path.append("./src")

from reg import infer_reg
from seg import ellipse_fitting
from seg.utils import load_infer_model

from tensorflow.keras.models import load_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str)
    parser.add_argument('--model_path', type=str, default="./models/regression_model.hdf5")
    parser.add_argument('--method', type=str, default='r',
                        help="'r'=regression, 's'=segmentation")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    image_path = args.image_path
    model_path = args.model_path

    if args.method == 'r':
        model = load_model(model_path, compile=False)
        infer_reg.show_pred(image_path, model)

    if args.method == 's':
        model = load_infer_model(model_path)
        ellipse_fitting.plot_pred(model, image_path)