import math
import keras
from dotenv import load_dotenv
import os
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from modules.aws import AWS
from modules.db_mlp import Db

load_dotenv()

class ResetLR(keras.callbacks.Callback):
    """ 
        This callback will reset the learning rate at each .fit call,
        Is meant to be used with ReduceLROnPlateau that will reduce the 
        learning rate if our val_loss is not improving but won't reset 
        the learning rate between .fit calls.
    """
    def __init__(self, lr = 1e-3):
        self.default_lr = lr

    def on_train_begin(self, logs=None):
        current_lr = self.model.optimizer.learning_rate
        if current_lr != self.default_lr:
            self.model.optimizer.learning_rate = self.default_lr

class MLPEegModel():
    
    def __init__(self, tune = False):

        tf.get_logger().setLevel('ERROR')

        self.aws = AWS()

        self.lr = 1e-3
        self.num_classes = 5
        self.epochs = 100
        self.tuner_epochs = 2
        self.batch_size = 64
        self.dir = os.getenv("MODEL_CHECKPOINT_DIR")
        self.db = Db()

        self.callbacks = [
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.3,
                patience=5,
                min_lr=1e-5,
                mode="min",
                verbose=1,
                cooldown=1
            ),
            ResetLR(
                lr=self.lr
            )
        ]

        if not tune:
            if os.path.exists(self.dir + "mlp_eeg.keras"):
                self.model = keras.models.load_model(self.dir + "mlp_eeg.keras", compile=True)
                # self.db.flushData()
            else:
                
                input = keras.Input(shape=(35,), name="EGGInput")
                hidden1 = keras.layers.Dense(100, activation="relu")(input)
                hidden2 = keras.layers.Dense(100, activation="relu")(hidden1)
                concat = keras.layers.Concatenate()([input,hidden2])         
                output = keras.layers.Dense(self.num_classes, activation="softmax")(concat)

                self.model =  keras.models.Model(inputs=input, outputs=output)

                self.model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=self.lr),
                    loss=keras.losses.CategoricalCrossentropy(),
                    metrics=[keras.metrics.CategoricalAccuracy()]
                )
                # self.db.restart()

            self.model.summary()

    def __tuneModel(self, hp):

        input = keras.Input(shape=(35,), name="EGGInput")
        hidden1 = keras.layers.Dense(hp.Choice('layer_1', [20,50,100]), activation="relu")(input)
        hidden2 = keras.layers.Dense(hp.Choice('layer_2', [20,50,100]), activation="relu")(hidden1)
        concat = keras.layers.Concatenate()([input,hidden2]) 
        output = keras.layers.Dense(self.num_classes, activation="softmax")(concat)

        model =  keras.models.Model(inputs=input, outputs=output, name = "EEGModel")
        learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])

        model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss=keras.losses.CategoricalCrossentropy(),
                metrics=[keras.metrics.CategoricalAccuracy()]
        )

        return model

    def tuneModel(self):
        tuner = kt.GridSearch(
            hypermodel=self.__tuneModel,
            objective="val_categorical_accuracy",
            executions_per_trial=1,
            overwrite=True,
            directory="out",
            project_name="mlp_tune",
        )

        def dataGen():
            while True:
                for x, y, w in self.db.readChunks(self.batch_size, self.tuner_epochs):
                    yield x, y, w

        def valGen():
            while True:
                for x, y, w in self.db.readChunks(self.batch_size, self.tuner_epochs, "validation"):
                    yield x, y, w

        tuner.search(
            dataGen(),
            steps_per_epoch=math.ceil(self.db.sampleNum() / self.batch_size),
            epochs=self.tuner_epochs,
            validation_data=valGen(),
            validation_steps=math.ceil(self.db.sampleNum("validation") / self.batch_size),
        )

        tuner.results_summary()

    def fit(self):

        history = self.model.fit(
            self.db.readChunks(self.batch_size, self.epochs),
            epochs=self.epochs,
            steps_per_epoch=math.ceil(self.db.sampleNum() / self.batch_size),
            validation_data=self.db.readChunks(self.batch_size, self.epochs, "validation"),
            validation_steps=math.ceil(self.db.sampleNum("validation") / self.batch_size),
            verbose=1,
            callbacks=[self.callbacks]
        )

        cat_acc = np.mean(history.history['categorical_accuracy'])
        val_cat_acc = np.mean(history.history['val_categorical_accuracy'])
        loss = np.mean(history.history['loss'])
        val_loss = np.mean(history.history['val_loss'])

        self.db.reconnect()

        return cat_acc, val_cat_acc, loss, val_loss

    def evaluate(self, chunk, tags):
        return self.model.evaluate(tf.stack(chunk), tf.stack(keras.utils.to_categorical(tags, num_classes=5)))
        
    def getMetrics(self):
        return self.model.metrics_names

    def save(self, chunks, inserted, mode = "ROWS", done = False, onlyChunks = False):

        if not onlyChunks:
            self.model.save(self.dir + "mlp_eeg.keras", include_optimizer=True)

        with open(self.dir + "mlp_eeg.chunks", "w") as chunksFile:
            chunksFile.write(f"{mode}={chunks}\n")
            chunksFile.write(f"INSERTED={inserted}\n")
            if done == True:
                chunksFile.write(f"DONE")