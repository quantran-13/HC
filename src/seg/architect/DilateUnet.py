import os
import sys
import numpy as np

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, ZeroPadding2D, Add, Multiply, Lambda
from tensorflow.keras import backend as K

from seg.utils import load_pretrain_model


def activation(x, batchnorm=True):
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def convolution_block(x, n_filters, kernel_size=3, strides=(1, 1), padding="same", dilation_rate=1, batchnorm=True):
    x = Conv2D(filters=n_filters,
               kernel_size=kernel_size,
               kernel_initializer="he_normal",
               strides=strides,
               padding=padding,
               dilation_rate=dilation_rate)(x)
    x = activation(x, batchnorm=batchnorm)

    return x


def conv2d_block(x, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    c1 = convolution_block(x,
                           n_filters,
                           kernel_size=kernel_size,
                           batchnorm=batchnorm)

    # second layer
    c2 = convolution_block(c1,
                           n_filters,
                           kernel_size=kernel_size,
                           batchnorm=batchnorm)

    return c2


def dilated_block(x, n_filters, kernel_size=3, batchnorm=True):
    """
        unet bottle neck with dilate convolution
    """
    d1 = convolution_block(x,
                           n_filters,
                           kernel_size=kernel_size,
                           dilation_rate=1,
                           batchnorm=batchnorm)
    d2 = convolution_block(d1,
                           n_filters,
                           kernel_size=kernel_size,
                           dilation_rate=2,
                           batchnorm=batchnorm)
    d3 = convolution_block(d2,
                           n_filters,
                           kernel_size=kernel_size,
                           dilation_rate=4,
                           batchnorm=batchnorm)
    d4 = convolution_block(d3,
                           n_filters,
                           kernel_size=kernel_size,
                           dilation_rate=8,
                           batchnorm=batchnorm)
    d5 = convolution_block(d4,
                           n_filters,
                           kernel_size=kernel_size,
                           dilation_rate=16,
                           batchnorm=batchnorm)
    d6 = convolution_block(d5,
                           n_filters,
                           kernel_size=kernel_size,
                           dilation_rate=32,
                           batchnorm=batchnorm)
    a = Add()([d1, d2, d3, d4, d5, d6])

    return a


def padding(x, y):
    x_shape = x.shape.as_list()
    y_shape = y.shape.as_list()

    if x.shape[1] > y_shape[1]:
        y = ZeroPadding2D(((1, 0), (0, 0)), data_format="channels_last")(y)
    if x.shape[1] < y_shape[1]:
        x = ZeroPadding2D(((1, 0), (0, 0)), data_format="channels_last")(x)
    if x.shape[2] > y_shape[2]:
        y = ZeroPadding2D(((0, 0), (1, 0)), data_format="channels_last")(y)
    if x.shape[2] < y_shape[2]:
        x = ZeroPadding2D(((0, 0), (1, 0)), data_format="channels_last")(x)

    return x, y


def dilate_unet(input_size=(216, 320, 1), n_filters=64, batchnorm=True, dropout_rate=0.1, freeze=False, freeze_at=0):
    inputs = Input(input_size, name="img")

    # contraction path
    c1 = conv2d_block(inputs, n_filters * 1,
                      kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout_rate)(p1)

    c2 = conv2d_block(p1, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout_rate)(p2)

    c3 = conv2d_block(p2, n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout_rate)(p3)

    c4 = dilated_block(p3, n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    # expansion path
    u5 = Conv2DTranspose(n_filters * 4, 3, strides=(2, 2), padding="same")(c4)
    u5 = concatenate([u5, c3])
    u5 = Dropout(dropout_rate)(u5)
    c5 = conv2d_block(u5, n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u6 = Conv2DTranspose(n_filters * 2, 3, strides=(2, 2), padding="same")(c5)
    u6 = concatenate([u6, c2])
    u6 = Dropout(dropout_rate)(u6)
    c6 = conv2d_block(u6, n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 1, 3, strides=(2, 2), padding="same")(c6)
    u7 = concatenate([u7, c1])
    u7 = Dropout(dropout_rate)(u7)
    c7 = conv2d_block(u7, n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(c7)
    model = Model(inputs=[inputs], outputs=[outputs],
                  name="DilateUNet")

    if freeze:
        fine_tune_at = freeze_at
        model_tmp = load_pretrain_model(
            "../models/model_dilate_unet.hdf5")

        for layer, layer_tmp in zip(model.layers[:fine_tune_at], model_tmp.layers[:fine_tune_at]):
            layer.set_weights(layer_tmp.get_weights())
            layer.trainable = False

    model.summary()
    # tf.keras.utils.plot_model(model, show_shapes=True)

    return model


if __name__ == "__main__":
    model = dilate_unet()
    dot_img_file = "../images/dilate_unet.png"
    tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
