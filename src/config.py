# data
PALETTE = [255]

# loss
LOSS = "bce_dice"
GAMMA = 2.0  # focal loss gamma

OPTIMIZER = "sgd"
METRICS = "dice_coeff"

# hyperparameter
DROPOUT_RATE = 0.1

IMAGE_SIZE = (216, 320, 1)
BATCH_SIZE = 16
EPOCHS = 5000

LEARNING_RATE = 0.1
MOMENTUM = 0.9
