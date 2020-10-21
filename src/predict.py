import os
import numpy as np
import pandas as pd
from PIL import Image

from config import *
from data import DataLoader
from utils import load_model_from_path


def eval(model_path):
    file_path = os.path.join("../models", model_path)
    model = load_model_from_path(file_path)

    print("Model train using 799 images and valid on 200 images.")
    valid_set = DataLoader("../data/training_set/", mode="valid", augmentation=True,
                           one_hot_encoding=True, palette=PALETTE, image_size=IMAGE_SIZE)
    valid_gen = valid_set.data_gen(BATCH_SIZE, shuffle=True)

    result = model.evaluate(valid_gen, verbose=0)
    print("Model score {} and loss {}".format(result[1], result[0]))


def predict(model_path, save_path="../data/predcited"):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    file_path = os.path.join("../models", model_path)
    model = load_model_from_path(file_path)

    print("="*100)
    print("LOADING TESTING DATA ...\n")
    test_set = DataLoader("../data/test_set/",
                          mode="test", image_size=IMAGE_SIZE)
    test_gen = test_set.data_gen(8)

    print("="*100)
    print("PREDICTING ...")

    preds_test = model.predict(test_gen)

    df = pd.read_csv("../data/test_set_pixel_size.csv")

    image_paths = [os.path.join(save_path, _[0])
                   for _ in df.values.tolist()]
    pre_paths = [_.replace(".png", "_Predicted_Mask.png") for _ in image_paths]

    for path, mask in zip(pre_paths, preds_test):
        image = Image.fromarray(mask.squeeze() * 255).convert("L")
        image.save(path)


if __name__ == "__main__":
    predict()
