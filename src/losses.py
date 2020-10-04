import numpy as np

from tensorflow.keras import backend as K


def jaccard_index(y_true, y_pred, smooth=1, reduction="mean"):
    y_true_f = K.abs(K.batch_flatten(y_true))
    y_pred_f = K.abs(K.batch_flatten(y_pred))

    intersection = K.sum(y_true_f * y_pred_f, axis=[1])
    union = K.sum(y_true_f, axis=[1]) + \
        K.sum(y_pred_f, axis=[1]) - intersection
    iou = (intersection + smooth) / (union + smooth)

    if reduction == "mean":
        return K.mean(iou)

    return iou


def jaccard_loss(y_true, y_pred, reduction="mean"):
    jaccard_score = jaccard_index(y_true, y_pred, reduction="none")
    loss = 1. - jaccard_score

    if reduction == "mean":
        return K.mean(loss)

    return loss


def dice_coeff(y_true, y_pred, smooth=1, reduction="mean"):
    y_true_f = K.abs(K.batch_flatten(y_true))
    y_pred_f = K.abs(K.batch_flatten(y_pred))

    intersection = K.sum(y_true_f * y_pred_f, axis=[1])
    union = K.sum(y_true_f, axis=[1]) + K.sum(y_pred_f, axis=[1])
    dice = (2. * intersection + smooth)/(union + smooth)

    if reduction == "mean":
        return K.mean(dice)

    return dice


def dice_loss(y_true, y_pred, reduction="mean"):
    dice_score = dice_coeff(y_true, y_pred, reduction="none")
    loss = 1. - dice_score

    if reduction == "mean":
        return K.mean(loss)

    return loss


if __name__ == "__main__":
    y_true = np.array([[[[1], [1], [1], [0], [0]],
                        [[1], [1], [1], [0], [0]],
                        [[1], [1], [1], [0], [0]],
                        [[1], [1], [1], [0], [0]],
                        [[1], [1], [1], [0], [0]]],

                       [[[1], [1], [1], [0], [0]],
                        [[1], [1], [1], [0], [0]],
                        [[1], [1], [1], [0], [0]],
                        [[1], [1], [1], [0], [0]],
                        [[1], [1], [1], [0], [0]]]])

    y_pred = np.array([[[[0], [0], [0], [1], [0]],
                        [[0], [0], [0], [1], [0]],
                        [[0], [0], [0], [1], [0]],
                        [[0], [0], [0], [1], [0]],
                        [[0], [0], [0], [1], [0]]],

                       [[[0], [0], [0], [1], [0]],
                        [[0], [0], [0], [1], [0]],
                        [[0], [0], [0], [1], [0]],
                        [[0], [0], [0], [1], [0]],
                        [[0], [0], [0], [1], [0]]]])

    print("jaccard_index: ", jaccard_index(
        K.variable(y_true), K.variable(y_pred)).numpy())
    print("jaccard_loss: ", jaccard_loss(
        K.variable(y_true), K.variable(y_pred)).numpy())
    print("dice_coeff: ", dice_coeff(K.variable(
        y_true), K.variable(y_pred)).numpy())
    print("dice_loss: ", dice_loss(K.variable(
        y_true), K.variable(y_pred)).numpy())
