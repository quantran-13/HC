# %%
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
from tensorflow_addons.losses import sigmoid_focal_crossentropy

# %%


def apply_reduction(value, reduction="mean"):
    if reduction == "mean":
        return K.mean(value)
    if reduction == "sum":
        return K.sum(value)

    return value


def jaccard_index(y_true, y_pred, smooth=1, reduction="mean"):
    """
        JI = TP / (TP + FP + FN)
    """
    y_true_f = K.abs(K.batch_flatten(y_true))
    y_pred_f = K.abs(K.batch_flatten(y_pred))

    intersection = K.sum(y_true_f * y_pred_f, axis=[1])
    union = K.sum(y_true_f, axis=[1]) + \
        K.sum(y_pred_f, axis=[1]) - intersection
    iou = (intersection + smooth) / (union + smooth)

    return apply_reduction(iou, reduction=reduction)


def jaccard_loss(y_true, y_pred, reduction="mean"):
    jaccard_score = jaccard_index(y_true, y_pred, reduction="none")
    loss = 1. - jaccard_score

    return apply_reduction(loss, reduction=reduction)


def dice_coeff(y_true, y_pred, smooth=1, reduction="mean"):
    """
        DSC = 2TP / (2TP + FN + FP)
    """
    y_true_f = K.abs(K.batch_flatten(y_true))
    y_pred_f = K.abs(K.batch_flatten(y_pred))

    intersection = K.sum(y_true_f * y_pred_f, axis=[1])
    union = K.sum(y_true_f, axis=[1]) + K.sum(y_pred_f, axis=[1])
    dice = (2. * intersection + smooth)/(union + smooth)

    return apply_reduction(dice, reduction=reduction)


def dice_loss(y_true, y_pred, reduction="mean"):
    dice_score = dice_coeff(y_true, y_pred, reduction="none")
    loss = 1. - dice_score

    return apply_reduction(loss, reduction=reduction)


def bce_loss(y_true, y_pred, reduction="mean"):
    """
        BCE(p, 'p) = -(p * log('p) + (1 - p) * log(1 - 'p)
    """
    loss = binary_crossentropy(y_true, y_pred)

    return apply_reduction(loss, reduction=reduction)


def bce_dice_loss(y_true, y_pred, loss_weight=(1., 1.), reduction="mean"):
    loss = loss_weight[0] * bce_loss(y_true, y_pred, reduction=reduction) + \
        loss_weight[1] * dice_loss(y_true, y_pred, reduction=reduction)

    return loss


def focal_loss(gamma=2.):
    def loss(y_true, y_pred, reduction="mean"):
        fl = sigmoid_focal_crossentropy(y_true, y_pred, gamma=gamma)

        return apply_reduction(fl, reduction=reduction)

    return loss


def focal_dice_loss(gamma=2., loss_weight=(1., 1.)):
    def f_d_loss(y_true, y_pred, loss_weight=loss_weight, reduction="mean"):
        fl = focal_loss(gamma=gamma)
        loss = loss_weight[0] * fl(y_true, y_pred, reduction=reduction) + \
            loss_weight[1] * dice_loss(y_true, y_pred, reduction=reduction)

        return loss

    return f_d_loss


# %%
if __name__ == "__main__":
    y_true = K.variable(np.array([[[[1], [1], [1], [0], [0]],
                                   [[1], [1], [1], [0], [0]],
                                   [[1], [1], [1], [0], [0]],
                                   [[1], [1], [1], [0], [0]],
                                   [[1], [1], [1], [0], [0]]],

                                  [[[1], [1], [1], [0], [0]],
                                   [[1], [1], [1], [0], [0]],
                                   [[1], [1], [1], [0], [0]],
                                   [[1], [1], [1], [0], [0]],
                                   [[1], [1], [1], [0], [0]]]]))

    y_pred = K.variable(np.array([[[[0], [0], [1], [1], [0]],
                                   [[0], [0], [1], [1], [0]],
                                   [[0], [0], [1], [1], [0]],
                                   [[0], [0], [1], [1], [0]],
                                   [[0], [0], [1], [1], [0]]],

                                  [[[0], [0], [1], [1], [0]],
                                   [[0], [0], [1], [1], [0]],
                                   [[0], [0], [1], [1], [0]],
                                   [[0], [0], [1], [1], [0]],
                                   [[0], [0], [1], [1], [0]]]]))

    print("jaccard_index: ", jaccard_index(y_true, y_pred).numpy())
    print("jaccard_loss: ", jaccard_loss(y_true, y_pred).numpy())
    print("dice_coeff: ", dice_coeff(y_true, y_pred).numpy())
    print("dice_loss: ", dice_loss(y_true, y_pred).numpy())
    fl = focal_loss()
    print("focal_loss: ", fl(y_true, y_pred).numpy())

# %%
