config = {
    "palette": [255],
    "loss": "bce_dice",
    "gamma": 2.0,  # focal loss gamma
    "optimizer": "sgd",
    "freeze": False,
    "freeze_at": 24,
    "dropout_rate": 0.1,
    "momentum": 0.9,
    "learning_rate": 0.1,
    # (216, 320, 1) # (270, 400, 1) # (432, 640, 1)
    "image_size": (216, 320, 1),
    "batch_size": 16,
    "epochs": 200
}
