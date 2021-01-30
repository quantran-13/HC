from tensorflow.keras.models import load_model
from seg.utils import load_infer_model
from seg import predict
from reg import infer_reg
import argparse

import sys
sys.path.append("./src")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str)
    parser.add_argument('mask_path', type=str, default=None)
    parser.add_argument('--model_path', type=str,
                        default="./models/regression_model.hdf5")
    parser.add_argument('--method', type=str, default='r',
                        help="'r': regression, 's': segmentation")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    image_path = args.image_path
    mask_path = args.mask_path
    model_path = args.model_path

    if args.method == 'r':
        model = load_model(model_path, compile=False)
        infer_reg.show_pred(image_path, model)

    if args.method == 's':
        model = load_infer_model(model_path)
        predict.plot_pred(model, image_path, mask_path)
