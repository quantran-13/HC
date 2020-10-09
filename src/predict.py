import os

from utils import load_model
from config import *
from data import DataLoader


def eval(model):
    print("Model train using 799 images and valid on 200 images.")
    valid_set = DataLoader("../data/training_set/",
                           augmentation=True, mode="valid", image_size=IMAGE_SIZE)
    valid_gen = valid_set.data_gen(BATCH_SIZE, shuffle=True)

    result = model.evaluate(valid_gen)
    print("Model score {} and loss {}".format(result[1], result[0]))


def predict(model_path):
    file_path = os.path.join("../models", model_path)
    model = load_model(file_path)

    eval(model)

    


if __name__ == "__main__":
    predict()
