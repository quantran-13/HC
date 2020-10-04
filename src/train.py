import os
from tqdm import tqdm
import datetime

import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from unet import unet
from config import *
import losses
from utils import time_to_timestr
from data import DataLoader


def train():
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)

    print("Epochs: {},\nBatch size: {},\nInput size: {}".format(EPOCHS,
                                                                BATCH_SIZE,
                                                                IMAGE_SIZE))

    # Datasets
    train_set = DataLoader("../data/training_set/",
                           mode="train", image_size=IMAGE_SIZE)
    train_gen = train_set.data_gen(BATCH_SIZE, shuffle=True)
    valid_set = DataLoader("../data/training_set/",
                           mode="valid", image_size=IMAGE_SIZE)
    valid_gen = train_set.data_gen(BATCH_SIZE, shuffle=True)

    print("Training set size: ", len(train_gen))
    print("Valid set size : ", len(valid_gen))
    print("Total: ", len(train_gen) + len(valid_gen))

    model = unet(IMAGE_SIZE)
    print("Model: ", model._name)

    optimizer = SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM)
    print("Optimizer: ", optimizer._name)

    model.compile(optimizer=optimizer,
                  loss=losses.dice_loss,
                  metrics=losses.dice_coeff)

    anne = ReduceLROnPlateau(monitor="dice_loss",
                             factor=0.2,
                             patience=10,
                             verbose=1,
                             min_lr=1e-5)

    early = EarlyStopping(monitor="val_dice_loss",
                          patience=10,
                          verbose=1)

    timestr = time_to_timestr()
    log_dir = "./logs/fit/{}".format(timestr)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, write_images=True)

    file_path = "./model/%s/%s_score={val_dice_loss:.2f}_%s_ep{epoch:02d}_bsize%d_insize%s_.hdf5" % (
        timestr,
        model._name,
        optimizer._name,
        BATCH_SIZE,
        IMAGE_SIZE
    )
    checkpoint = ModelCheckpoint(file_path, verbose=1, save_best_only=True)

    callbacks_list = [anne, checkpoint, early, tensorboard_callback]

    history = model.fit(train_gen,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        callbacks=callbacks_list,
                        validation_data=valid_gen,
                        workers=4,
                        use_multiprocessing=True)


if __name__ == "__main__":
    train()
