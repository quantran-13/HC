import os
import sys
import numpy as np

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, ZeroPadding2D

from config import *


def conv2d_block(input, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters,
               kernel_size=kernel_size,
               kernel_initializer='he_normal',
               padding='same')(input)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # second layer
    x = Conv2D(filters=n_filters,
               kernel_size=kernel_size,
               kernel_initializer='he_normal',
               padding='same')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def unet(input_size=(256, 256, 1), n_filters=64, batchnorm=True):
    inputs = Input(input_size, name="img")

    # contraction path
    c1 = conv2d_block(inputs, n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(DROPOUT_RATE)(p1)

    c2 = conv2d_block(p1, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(DROPOUT_RATE)(p2)

    c3 = conv2d_block(p2, n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(DROPOUT_RATE)(p3)

    c4 = conv2d_block(p3, n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(DROPOUT_RATE)(p4)

    c5 = conv2d_block(p4, n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # expansion path
    u6 = Conv2DTranspose(n_filters * 8, 3, strides=(2, 2), padding='same')(c5)
    u6 = ZeroPadding2D(((1, 0), (0, 0)), data_format="channels_last")(u6)
    u6 = concatenate([u6, c4])
    u6 = Dropout(DROPOUT_RATE)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, 3, strides=(2, 2), padding='same')(c6)
    # u7 = ZeroPadding2D(((1, 0), (0, 0)), data_format="channels_last")(u7)
    u7 = concatenate([u7, c3])
    u7 = Dropout(DROPOUT_RATE)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, 3, strides=(2, 2), padding='same')(c7)
    # u8 = ZeroPadding2D(((1, 0), (0, 0)), data_format="channels_last")(u8)
    u8 = concatenate([u8, c2])
    u8 = Dropout(DROPOUT_RATE)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, 3, strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(DROPOUT_RATE)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    c10 = Conv2D(2, 3, activation='relu',
                 kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c10)
    model = Model(inputs=[inputs], outputs=[outputs], name="UNet")

    # model.summary()
    # tf.keras.utils.plot_model(model, show_shapes=True)

    return model


if __name__ == "__main__":
    model = unet(input_size=IMAGE_SIZE)
