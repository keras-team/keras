'''collection of messages which are sent between a model and its callbacks
'''

from rx.subjects import Subject

class Message(object):
    pass

# events emitted from model
class BatchBegin(Message):
    def __init__(self, batch, size):
        self.batch = batch
        self.size = size

class BatchEnd(Message):
    def __init__(self, batch, size, accuracy, loss):
        self.batch = batch
        self.size = size
        if accuracy:
            self.accuracy = accuracy
        else:
            self.accuracy = 0.
        if loss:
            self.loss = loss
        else:
            self.loss = 0.

class EpochEnd(Message):
    def __init__(self, epoch, val_accuracy, val_loss):
        self.epoch = epoch
        if val_accuracy:
            self.val_accuracy = val_accuracy
        else:
            self.val_accuracy = 0.
        if val_loss:
            self.val_loss = val_loss
        else:
            self.val_loss = 0.

class EpochBegin(Message):
    def __init__(self, epoch):
        self.epoch = epoch

class TrainBegin(Message):
    def __init__(self):
        pass

class TrainEnd(Message):
    def __init__(self):
       pass

# these are commands sent to the model
# to stop model fitting
class StopTraining(Message):
    pass
# to save the current weights
class SaveModel(Message):
    def __init__(self, filename, overwrite):
        self.filename = filename
        self.overwrite = overwrite

