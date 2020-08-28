import time
import keras.callbacks


class TimeHistory(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.times = []

    def on_train_begin(self, logs={}):
        pass

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
