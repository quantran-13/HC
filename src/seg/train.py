import os
import numpy as np
import pandas as pd

import math
import datetime
from tqdm import tqdm
from pathlib import Path

from seg import seglosses
from seg.config import config
from seg.data import DataLoader
from seg.utils import time_to_timestr
from seg.SGDRScheduler import SGDRScheduler

from seg.architect.Unet import unet
from seg.architect.DilateUnet import dilate_unet
from seg.architect.AttentionUnet import attention_unet
from seg.architect.DilateAttentionUnet import dilate_attention_unet

import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.keras.optimizers.schedules import InverseTimeDecay
tf.get_logger().setLevel("INFO")


def lr_step_decay(epoch, lr):
    """
    Step decay lr: learning_rate = initial_lr * drop_rate^floor(epoch / epochs_drop)
    """
    initial_learning_rate = config["learning_rate"]
    drop_rate = 0.5
    epochs_drop = 10.0

    return initial_learning_rate * math.pow(drop_rate, math.floor(epoch/epochs_drop))


def train():

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)

    print("Epochs: {}\t\tBatch size: {}\t\tInput size: {}".format(config["epochs"],
                                                                  config["batch_size"],
                                                                  config["image_size"]))

    # Datasets
    print("="*100)
    print("LOADING DATA ...\n")
    train_set = DataLoader("../data/training_set/",
                           mode="train",
                           augmentation=True,
                           compose=False,
                           one_hot_encoding=True,
                           palette=config["palette"],
                           image_size=config["image_size"])
    train_gen = train_set.data_gen(config["batch_size"], shuffle=True)

    valid_set = DataLoader("../data/training_set/",
                           mode="valid",
                           augmentation=True,
                           compose=False,
                           one_hot_encoding=True,
                           palette=config["palette"],
                           image_size=config["image_size"])
    valid_gen = valid_set.data_gen(config["batch_size"], shuffle=True)

    # define model
    model = dilate_unet(input_size=config["image_size"],
                        dropout_rate=config["dropout_rate"],
                        freeze=config["freeze"],
                        freeze_at=config["freeze_at"])
    print("Model: ", model._name)

    # optim
    optimizers = {
        "sgd": SGD(learning_rate=config["learning_rate"], momentum=config["momentum"], nesterov=True),
        "adam": Adam(learning_rate=config["learning_rate"], amsgrad=True),
        "rmsprop": RMSprop(learning_rate=config["learning_rate"], momentum=config["momentum"])
    }

    optimizer = optimizers[config["optimizer"]]
    print("Optimizer: ", optimizer._name)

    # loss
    losses = {
        "jaccard": seglosses.jaccard_loss,
        "dice": seglosses.dice_loss,
        "bce": seglosses.bce_loss,
        "bce_dice": seglosses.bce_dice_loss,
        "focal": seglosses.focal_loss(gamma=config["gamma"]),
        "focal_dice": seglosses.focal_dice_loss(gamma=config["gamma"]),
    }
    print("Loss: ", losses[config["loss"]])

    model.compile(optimizer=optimizer,
                  loss=[losses[config["loss"]]],
                  metrics=[seglosses.jaccard_index, seglosses.dice_coeff, seglosses.bce_loss])

    # callbacks
    lr_schedule = SGDRScheduler(min_lr=1e-5,
                                max_lr=config["learning_rate"],
                                steps_per_epoch=np.ceil(
                                    config["epochs"] / config["batch_size"]),
                                lr_decay=0.9,
                                cycle_length=10,
                                mult_factor=1.5)

    # lr_schedule = LearningRateScheduler(lr_step_decay,
    #                                     verbose=1)

    anne = ReduceLROnPlateau(monitor="loss",
                             factor=0.2,
                             patience=20,
                             verbose=1,
                             min_lr=1e-7)

    early = EarlyStopping(monitor="val_loss",
                          patience=50,
                          verbose=1)

    timestr = time_to_timestr()
    log_dir = "../logs/fit/{}".format(timestr)
    tensorboard_callback = TensorBoard(log_dir=log_dir,
                                       write_images=True)

    Path("../models/{}".format(timestr)).mkdir(parents=True, exist_ok=True)
    file_path = "../models/%s/%s_%s_ep{epoch:02d}_bsize%d_insize%s.hdf5" % (
        timestr,
        model._name,
        optimizer._name,
        config["batch_size"],
        config["image_size"]
    )
    checkpoint = ModelCheckpoint(file_path, verbose=1, save_best_only=True)

    callbacks_list = [
        lr_schedule,
        early,
        # anne,
        checkpoint,
        tensorboard_callback
    ]

    print("="*100)
    print("TRAINING ...\n")

    history = model.fit(train_gen,
                        batch_size=config["batch_size"],
                        epochs=config["epochs"],
                        callbacks=callbacks_list,
                        validation_data=valid_gen,
                        workers=8,
                        use_multiprocessing=True)

    his = pd.DataFrame(history.history)
    his.to_csv("../models/{}/history.csv".format(timestr), index=False)

    his = pd.DataFrame(lr_schedule.history)
    his.to_csv("../models/{}/history_lr.csv".format(timestr), index=False)

    print("="*100)


if __name__ == "__main__":
    train()
