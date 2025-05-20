import math
import keras
from dotenv import load_dotenv
import os
import numpy as np
import tensorflow as tf
from sklearn.utils import gen_batches, shuffle
import keras_tuner as kt
from modules.db import Db

load_dotenv()
class EEGModel():
    
    def __init__(self, tune = False):

        # https://arxiv.org/abs/1611.06455
        # https://keras.io/examples/timeseries/timeseries_classification_from_scratch/

        tf.get_logger().setLevel('ERROR')
        self.num_classes = 5

        self.db = Db()

        self.epochs = 5
        self.tuner_epochs = 2
        self.batch_size = 64
        self.dir = os.getenv("MODEL_CHECKPOINT_DIR")

        if not tune:
            if os.path.exists(self.dir + "eeg.keras"):
                self.model = keras.models.load_model(self.dir + "eeg.keras", compile=True)
                # self.db.flushData()
            else:
                
                input = keras.Input(shape=(6001, 18), name="EGGInput")

                conv1 = keras.layers.Conv1D(filters=32, kernel_size=3, padding="same")(input)
                conv1 = keras.layers.BatchNormalization()(conv1)
                conv1 = keras.layers.ReLU()(conv1)

                conv2 = keras.layers.Conv1D(filters=32, kernel_size=5, padding="same")(conv1)
                conv2 = keras.layers.BatchNormalization()(conv2)
                conv2 = keras.layers.ReLU()(conv2)

                conv3 = keras.layers.Conv1D(filters=64, kernel_size=5, padding="same")(conv2)
                conv3 = keras.layers.BatchNormalization()(conv3)
                conv3 = keras.layers.ReLU()(conv3)

                gap = keras.layers.GlobalAveragePooling1D()(conv3)

                output = keras.layers.Dense(self.num_classes, activation="softmax")(gap)

                self.model =  keras.models.Model(inputs=input, outputs=output)

                self.model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
                    loss=keras.losses.CategoricalCrossentropy(),
                    metrics=[keras.metrics.CategoricalAccuracy()]
                )
                #self.db.restart()

            self.model.summary()

    def __tuneModel(self, hp):

        input = keras.Input(shape=(6001, 18))

        conv1 = keras.layers.Conv1D(filters=hp.Choice('filters_1', [32, 64, 128, 256]), kernel_size=hp.Choice('kernel_size_1', [3,5,7]), padding="same")(input)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.ReLU()(conv1)

        conv2 = keras.layers.Conv1D(filters=hp.Choice('filters_2', [32, 64, 128, 256]), kernel_size=hp.Choice('kernel_size_2', [3,5,7]), padding="same")(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.ReLU()(conv2)

        conv3 = keras.layers.Conv1D(filters=hp.Choice('filters_3', [32, 64, 128, 256]), kernel_size=hp.Choice('kernel_size_3', [3,5,7]), padding="same")(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.ReLU()(conv3)

        gap = keras.layers.GlobalAveragePooling1D()(conv3)

        output = keras.layers.Dense(self.num_classes, activation="softmax")(gap)

        model =  keras.models.Model(inputs=input, outputs=output, name = "EEGModel")
        learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4, 1e-5])

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
            project_name="eeg_tune",
        )

        def dataGen():
            while True:
                for x, y, w in self.db.readChunks(self.batch_size, self.tuner_epochs):
                    yield x, y, w

        def valGen():
            while True:
                for x, y, w in self.db.readChunks(self.batch_size, self.tuner_epochs, "validation"):
                    yield x, y, w

        print(type(dataGen), type(valGen))

        tuner.search(
            dataGen(),
            steps_per_epoch=math.ceil(self.db.sampleNum() / self.batch_size),
            epochs=self.tuner_epochs,
            validation_data=valGen(),
            validation_steps=math.ceil(self.db.sampleNum("validation_tags") / self.batch_size),
        )

        tuner.results_summary()

    def getModel(self):
        return self.model

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

    def plot(self):
        keras.utils.plot_model(self.model, to_file="memoria/figs/fcn.png", show_shapes=True)