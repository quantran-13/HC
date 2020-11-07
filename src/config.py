# data
PALETTE = [255]

# loss
LOSS = "bce_dice"
GAMMA = 2.0  # focal loss gamma

OPTIMIZER = "sgd"

FREEZE = False
FREEZE_AT = 0

# hyperparameter
DROPOUT_RATE = 0.1
MOMENTUM = 0.9
LEARNING_RATE = 1e-3

IMAGE_SIZE = (216, 320, 1)
BATCH_SIZE = 32
EPOCHS = 20
