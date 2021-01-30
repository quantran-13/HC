import os
import pandas as pd

from PIL import Image
from pathlib import Path

import tensorflow as tf

from seg.config import config
from seg.data import DataLoader, load_infer_image
from seg.utils import load_infer_model


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


if __name__ == "__main__":
    pred()
