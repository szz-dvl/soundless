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
    
    def __init__(self):

        tf.get_logger().setLevel('ERROR')

        self.db = Db()
        self.lr = 0.01
        self.optimizer = keras.optimizers.SGD(learning_rate=self.lr, momentum=0.9, nesterov=True)
        self.callbacks = [
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.1,
                patience=2,
                min_lr=1e-5,
                mode="min"
            )
        ]
        
        self.epochs = 5
        self.batch_size = 64
        self.dir = os.getenv("MODEL_CHECKPOINT_DIR")

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

    def fit(self):
        
        history = self.model.fit(
            self.db.readChunks(self.batch_size, self.epochs),
            epochs=self.epochs,
            steps_per_epoch=math.ceil(self.db.sampleNum() / self.batch_size),
            validation_data=self.db.readChunks(self.batch_size, self.epochs, "validation"),
            validation_steps=math.ceil(self.db.sampleNum("validation_tags") / self.batch_size),
            callbacks=[self.callbacks],
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