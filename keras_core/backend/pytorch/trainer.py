from keras_core.trainers import trainer as base_trainer


class PyTorchTrainer(base_trainer.Trainer):
    def __init__(self):
        super().__init__()
