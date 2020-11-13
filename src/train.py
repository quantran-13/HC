import os
import datetime
import pandas as pd
from tqdm import tqdm
import math

import seglosses
from config import *
from unet import unet
from architect.DilateAttentionUnet import dilate_attention_unet
from data import DataLoader
from utils import time_to_timestr

import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.keras.optimizers.schedules import InverseTimeDecay
tf.get_logger().setLevel("INFO")


def lr_step_decay(epoch, lr):
    """
    Step decay lr: learning_rate = initial_lr * drop_rate^floor(epoch / epochs_drop)
    """
    initial_learning_rate = LEARNING_RATE
    drop_rate = 0.5
    epochs_drop = 10.0

    return initial_learning_rate * math.pow(drop_rate, math.floor(epoch/epochs_drop))


def train():
    if not os.path.exists("../models"):
        os.mkdir("../models")

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)

    print("Epochs: {}\t\tBatch size: {}\t\tInput size: {}".format(EPOCHS,
                                                                  BATCH_SIZE,
                                                                  IMAGE_SIZE))

    # Datasets
    print("="*100)
    print("LOADING DATA ...\n")
    train_set = DataLoader("../data/training_set/",
                           mode="train",
                           augmentation=True,
                           one_hot_encoding=True,
                           palette=PALETTE,
                           image_size=IMAGE_SIZE)
    train_gen = train_set.data_gen(BATCH_SIZE, shuffle=True)

    valid_set = DataLoader("../data/training_set/",
                           mode="valid",
                           augmentation=True,
                           one_hot_encoding=True,
                           palette=PALETTE,
                           image_size=IMAGE_SIZE)
    valid_gen = valid_set.data_gen(BATCH_SIZE, shuffle=True)

    # define model
    model = dilate_attention_unet(dropout_rate=DROPOUT_RATE,
                                  freeze=FREEZE,
                                  freeze_at=FREEZE_AT)
    print("Model: ", model._name)

    # optim
    optimizers = {
        "sgd": SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM, nesterov=True),
        "adam": Adam(learning_rate=LEARNING_RATE, amsgrad=True),
        "rmsprop": RMSprop(learning_rate=LEARNING_RATE, momentum=MOMENTUM)
    }

    optimizer = optimizers[OPTIMIZER]
    print("Optimizer: ", optimizer._name)

    # loss
    loss = {
        "jaccard": seglosses.jaccard_loss,
        "dice": seglosses.dice_loss,
        "bce": seglosses.bce_loss,
        "bce_dice": seglosses.bce_dice_loss,
        "focal": seglosses.focal_loss(gamma=GAMMA)
    }

    model.compile(optimizer=optimizer,
                  loss=[loss[LOSS]],
                  metrics=[seglosses.jaccard_index, seglosses.dice_coeff, seglosses.bce_loss])

    # callbacks
    lr_schedule = LearningRateScheduler(lr_step_decay,
                                        verbose=1)

    anne = ReduceLROnPlateau(monitor="loss",
                             factor=0.2,
                             patience=30,
                             verbose=1,
                             min_lr=1e-7)

    early = EarlyStopping(monitor="val_loss",
                          patience=50,
                          verbose=1)

    timestr = time_to_timestr()
    log_dir = "../logs/fit/{}".format(timestr)
    tensorboard_callback = TensorBoard(log_dir=log_dir,
                                       write_images=True)

    os.mkdir("../models/{}".format(timestr))
    file_path = "../models/%s/%s_%s_ep{epoch:02d}_bsize%d_insize%s.hdf5" % (
        timestr,
        model._name,
        optimizer._name,
        BATCH_SIZE,
        IMAGE_SIZE
    )
    checkpoint = ModelCheckpoint(file_path, verbose=1, save_best_only=True)

    callbacks_list = [
        lr_schedule,
        early,
        anne,
        checkpoint,
        tensorboard_callback
    ]

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
