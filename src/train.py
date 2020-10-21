from utils import time_to_timestr
from data import DataLoader
from unet import unet
from config import *
import losses
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
import datetime
import pandas as pd
from tqdm import tqdm

import tensorflow as tf
tf.get_logger().setLevel('INFO')


def train():
    if not os.path.exists("../models"):
        os.mkdir("../models")

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
    valid_gen = valid_set.data_gen(BATCH_SIZE, shuffle=True)

    model = unet(IMAGE_SIZE)
    print("Model: ", model._name)

    optimizer = SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM)
    print("Optimizer: ", optimizer._name)

    model.compile(optimizer=optimizer,
                  loss=[losses.jaccard_loss],
                  metrics=[losses.jaccard_index])

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

    history = model.fit(train_gen,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        callbacks=callbacks_list,
                        validation_data=valid_gen,
                        workers=4,
                        use_multiprocessing=True)

    his = pd.DataFrame(history.history)
    his.to_csv("../models/{}/history.csv".format(timestr), index=False)


if __name__ == "__main__":
    train()
