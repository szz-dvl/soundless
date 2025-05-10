import math
import keras
from keras import layers
from dotenv import load_dotenv
import os
import numpy as np
import tensorflow as tf
from sklearn.utils import gen_batches, shuffle

from modules.db import Db

load_dotenv()

class EEGModel():
    
    def __init__(self, outdir = "out/"):

        tf.get_logger().setLevel('ERROR')

        self.db = Db()
        self.lr = 0.001
        self.lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.lr,
            decay_steps=1000,
            decay_rate=0.6,
            staircase=True
        )
        self.optimizer = keras.optimizers.AdamW(learning_rate=self.lr_schedule)
        self.epochs = 5
        self.batch_size = 64
        self.outdir = outdir
        self.dir = os.getenv("MODEL_CHECKPOINT_DIR")
        self.lr_cb = keras.callbacks.LearningRateScheduler(self.__lrDecayCb, verbose=False)

        if os.path.exists(self.dir + "eeg.keras"):
            self.model = keras.models.load_model(self.dir + "eeg.keras", compile=True)
            self.db.flushData()
        else:
            input = keras.Input(shape=(6001, 18), name="EGGInput")
            x = layers.Conv1D(64, (3), strides=(2), use_bias=False, name="EEGConv")(input)
            x = layers.LayerNormalization(scale=False, center=True, name="EGGScaler")(x)
            x = layers.Activation(activation="relu", name="EGGReLU")(x)
            x = layers.MaxPooling1D(pool_size=(2), name="EEGMaxPooling")(x)
            x = layers.Dropout(0.2, name="EGGDropoutPre")(x)
            x = layers.LSTM(100, name="EEGLstm")(x)
            x = layers.Dropout(0.2, name="EGGDropoutPost")(x)
            outputs = layers.Dense(5, activation="softmax", name="EGGOutput")(x)
            self.model = keras.Model(inputs = input, outputs = outputs, name = "EEGModel")
            self.model.compile(
                optimizer=self.optimizer,
                loss=keras.losses.CategoricalCrossentropy(),
                metrics=[keras.metrics.CategoricalAccuracy()]
            )
            self.db.restart()

        self.model.summary()

    def getModel(self):
        return self.model

    def __lrDecayCb(self, epoch):
        return self.lr * math.pow(0.6, epoch)

    def __lrDecay(self, epoch):
        self.optimizer.learning_rate.assign(self.lr * math.pow(0.6, epoch))

    def __resetLr(self):
        self.optimizer.learning_rate.assign(self.lr)

    def feed(self, chunk, tags) -> float:
        
        result = None
        for epoch in range(self.epochs):
            x, y = shuffle(chunk, tags)
            self.__lrDecay(epoch)
            for slice in gen_batches(len(x), self.batch_size):
                result = self.model.train_on_batch(
                    tf.stack(x[slice]),
                    tf.stack(keras.utils.to_categorical(y[slice], num_classes=5)),
                    return_dict=True
                )

        self.__resetLr()
        return result['categorical_accuracy']

    def fit(self):
        
        history = self.model.fit(
            self.db.readChunks(self.batch_size, self.epochs),
            epochs=self.epochs,
            steps_per_epoch=math.ceil(self.db.sampleNum() / self.batch_size),
            validation_data=self.db.readChunks(self.batch_size, self.epochs, "validation"),
            validation_steps=math.ceil(self.db.sampleNum("validation_tags") / self.batch_size),
            verbose=0
        )

        cat_acc = np.mean(history.history['categorical_accuracy'])
        val_cat_acc = np.mean(history.history['val_categorical_accuracy'])
        loss = np.mean(history.history['loss'])
        val_loss = np.mean(history.history['val_loss'])

        # For some reason DDBB gets stuck after training T_T
        self.db.reconnect()

        return cat_acc, val_cat_acc, loss, val_loss

    def evaluate(self, chunk, tags):
        return self.model.evaluate(tf.stack(chunk), tf.stack(keras.utils.to_categorical(tags, num_classes=5)))
        
    def getMetrics(self):
        return self.model.metrics_names

    def save(self, chunks, mode = "ROWS", done = False):
        self.model.save(self.dir + "eeg.keras", include_optimizer=True)
        with open(self.dir + "eeg.chunks", "w") as chunksFile:
            chunksFile.write(f"{mode}={chunks}\n")
            if done == True:
                chunksFile.write(f"DONE")