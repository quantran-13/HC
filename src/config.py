# data
PALETTE = [255]

# loss
LOSS = "bce_dice"
GAMMA = 2.0  # focal loss gamma

OPTIMIZER = "sgd"

FREEZE_AT = 16

# hyperparameter
DROPOUT_RATE = 0.1
MOMENTUM = 0.9
LEARNING_RATE = 1e-4

IMAGE_SIZE = (216, 320, 1)
BATCH_SIZE = 8
EPOCHS = 500
