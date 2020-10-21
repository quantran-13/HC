from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam
import os
import datetime
import pandas as pd
from tqdm import tqdm

import losses
from config import *
from unet import unet
from data import DataLoader
from utils import time_to_timestr

import tensorflow as tf
tf.get_logger().setLevel("INFO")


optimizers = {
    "sgd": SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM),
    "adam": Adam(learning_rate=LEARNING_RATE)
}

loss = {
    "jaccard": losses.jaccard_loss,
    "dice": losses.dice_loss
}

metrics = {
    "jaccard_index": losses.jaccard_index,
    "dice_coeff": losses.dice_coeff
}


def train():
    if not os.path.exists("../models"):
        os.mkdir("../models")

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)

    print("Epochs: {},\nBatch size: {},\nInput size: {}".format(EPOCHS,
                                                                BATCH_SIZE,
                                                                IMAGE_SIZE))

    # Datasets
    print("="*100)
    print("LOADING DATA ...\n")

    train_set = DataLoader("../data/training_set/", mode="train", augmentation=True,
                           one_hot_encoding=True, palette=PALETTE, image_size=IMAGE_SIZE)
    train_gen = train_set.data_gen(BATCH_SIZE, shuffle=True)
    valid_set = DataLoader("../data/training_set/", mode="valid", augmentation=True,
                           one_hot_encoding=True, palette=PALETTE, image_size=IMAGE_SIZE)
    valid_gen = valid_set.data_gen(BATCH_SIZE, shuffle=True)

    model = unet(IMAGE_SIZE)
    print("Model: ", model._name)

    optimizer = optimizers[OPTIMIZER]
    print("Optimizer: ", optimizer._name)

    model.compile(optimizer=optimizer,
                  loss=[loss[LOSS]],
                  metrics=[metrics[METRICS]])

    anne = ReduceLROnPlateau(monitor="loss",
                             factor=0.2,
                             patience=10,
                             verbose=1,
                             min_lr=1e-5)

    timestr = time_to_timestr()
    log_dir = "../logs/fit/{}".format(timestr)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, write_images=True)

    os.mkdir("../models/{}".format(timestr))
    file_path = "../models/%s/%s_jaccard={val_jaccard_index:.2f}_%s_ep{epoch:02d}_bsize%d_insize%s.hdf5" % (
        timestr,
        model._name,
        optimizer._name,
        BATCH_SIZE,
        IMAGE_SIZE
    )
    checkpoint = ModelCheckpoint(file_path, verbose=1, save_best_only=True)

    callbacks_list = [anne, checkpoint, tensorboard_callback]

    print("="*100)
    print("TRAINING ...\n")

    history = model.fit(train_gen,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        callbacks=callbacks_list,
                        validation_data=valid_gen,
                        workers=8,
                        use_multiprocessing=True)

    his = pd.DataFrame(history.history)
    his.to_csv("../models/{}/history.csv".format(timestr), index=False)


if __name__ == "__main__":
    train()
