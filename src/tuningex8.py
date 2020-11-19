import os
import datetime
import pandas as pd
from tqdm import tqdm
import math

import seglosses
from config import *
from data import DataLoader
from utils import time_to_timestr

from architect.Unet import unet
from architect.DilateAttentionUnet import dilate_attention_unet
from architect.AttentionUnet import attention_unet
from architect.DilateUnet import dilate_unet

import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.keras.optimizers.schedules import InverseTimeDecay
from tensorboard.plugins.hparams import api as hp
tf.get_logger().setLevel("INFO")


# HP_FREEZE_AT = hp.HParam("freeze_at", hp.Discrete([16, 24]))
# HP_DROPOUT = hp.HParam("dropout", hp.RealInterval(0.1, 0.2))
# HP_OPTIMIZER = hp.HParam("optimizer", hp.Discrete(["sgd", "adam", "rmsprop"]))
HP_LOSS = hp.HParam("loss", hp.Discrete(
    ["bce_dice", "jaccard", "dice", "focal"]))


HPARAMS = [
    HP_LOSS,
]

METRICS = [
    hp.Metric("epoch_jaccard_index",
              group="validation",
              display_name="Jaccard Index (val.)"),
    hp.Metric("epoch_dice_coeff",
              group="validation",
              display_name="Dice Coeff (val.)")
]


def lr_step_decay(epoch, lr):
    """
    Step decay lr: learning_rate = initial_lr * drop_rate^floor(epoch / epochs_drop)
    """
    initial_learning_rate = LEARNING_RATE
    drop_rate = 0.5
    epochs_drop = 10.0

    return initial_learning_rate * math.pow(drop_rate, math.floor(epoch/epochs_drop))


def train(run_dir, hparams, train_gen, valid_gen):
    # define model
    model = attention_unet(dropout_rate=DROPOUT_RATE,
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
    losses = {
        "jaccard": seglosses.jaccard_loss,
        "dice": seglosses.dice_loss,
        "bce": seglosses.bce_loss,
        "bce_dice": seglosses.bce_dice_loss,
        "focal": seglosses.focal_loss(gamma=GAMMA)
    }
    loss = losses[hparams[HP_LOSS]]
    print("Loss: ", loss)

    model.compile(optimizer=optimizer,
                  loss=[loss],
                  metrics=[seglosses.jaccard_index, seglosses.dice_coeff, seglosses.bce_loss])

    # callbacks
    lr_schedule = LearningRateScheduler(lr_step_decay,
                                        verbose=1)

    anne = ReduceLROnPlateau(monitor="loss",
                             factor=0.2,
                             patience=15,
                             verbose=1,
                             min_lr=1e-7)

    early = EarlyStopping(monitor="val_loss",
                          patience=50,
                          verbose=1)

    tensorboard_callback = TensorBoard(log_dir=run_dir,
                                       write_images=True)
    hparams_callback = hp.KerasCallback(run_dir, hparams)

    file_path = "../models/%s/%s/%s_%s_ep{epoch:02d}_bsize%d_insize%s.hdf5" % (
        run_dir.split("/")[-2],
        run_dir.split("/")[-1],
        model._name,
        optimizer._name,
        BATCH_SIZE,
        IMAGE_SIZE
    )
    os.mkdir("../models/{}/{}".format(run_dir.split("/")[-2],
                                      run_dir.split("/")[-1]))
    checkpoint = ModelCheckpoint(file_path, verbose=1, save_best_only=True)

    callbacks_list = [
        lr_schedule,
        early,
        anne,
        checkpoint,
        tensorboard_callback,
        hparams_callback
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
    his.to_csv("../models/{}/{}/history.csv".format(run_dir.split("/")[-2],
                                                    run_dir.split("/")[-1]), index=False)


def prepare_data():
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

    return (train_gen, valid_gen)


def main():
    if not os.path.exists("../models"):
        os.mkdir("../models")

    print("Epochs: {}\t\tBatch size: {}\t\tInput size: {}".format(EPOCHS,
                                                                  BATCH_SIZE,
                                                                  IMAGE_SIZE))

    # Datasets
    print("="*100)
    print("LOADING DATA ...\n")
    (train_gen, valid_gen) = prepare_data()

    timestr = time_to_timestr()
    os.mkdir("../models/{}".format(timestr))
    log_dir = "../logs/fit/{}".format(timestr)

    with tf.summary.create_file_writer(log_dir).as_default():
        hp.hparams_config(hparams=HPARAMS, metrics=METRICS)

    session_num = 0
    for loss in HP_LOSS.domain.values:
        hparams = {
            HP_LOSS: loss
        }

        run_name = "run-{}".format(session_num)
        print("---- Starting trial: {} ----".format(run_name))
        print({h.name: hparams[h] for h in hparams})
        train("{}/{}".format(log_dir, run_name),
              hparams,
              train_gen,
              valid_gen)
        session_num += 1


if __name__ == "__main__":
    main()
