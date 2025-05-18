import math
import keras
from dotenv import load_dotenv
import os
import numpy as np
import tensorflow as tf
from sklearn.utils import gen_batches, shuffle
import keras_tuner as kt
from modules.aws import AWS
from modules.chann_selector import ChannSelector

load_dotenv()

class MLPEegModel():
    
    def __init__(self):

        tf.get_logger().setLevel('ERROR')

        self.aws = AWS()

        self.num_classes = 5
        self.epochs = 5
        self.tuner_epochs = 2
        self.batch_size = 64
        self.dir = os.getenv("MODEL_CHECKPOINT_DIR")

        self.callbacks = [
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.1,
                patience=2,
                min_lr=1e-5,
                mode="min"
            )
        ]

        self.validation_set = {
                "folder": "sub-S0001111192396",
                "session": 1,
                "site": "S0001"
        }

        self.validation_data = None

        self.__getValidationData()

        if os.path.exists(self.dir + "mlp_eeg.keras"):
            self.model = keras.models.load_model(self.dir + "eeg.keras", compile=True)
        else:
            
            input = keras.Input(shape=(35,), name="EGGInput")
            x = keras.layers.Dense(100, activation="relu")(input)
            x = keras.layers.Dense(50, activation="relu")(x)
            output = keras.layers.Dense(self.num_classes, activation="softmax")(x)

            self.model =  keras.models.Model(inputs=input, outputs=output)

            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                loss=keras.losses.CategoricalCrossentropy(),
                metrics=[keras.metrics.CategoricalAccuracy()]
            )

        self.model.summary()

    def __tuneModel(self, hp):

        input = keras.Input(shape=(35,), name="EGGInput")
        x = keras.layers.Dense(hp.Choice('dense_1', [20, 50, 100]), activation="relu")(input)
        x = keras.layers.Dense(hp.Choice('dense_2', [10, 30, 50]), activation="relu")(x)
        output = keras.layers.Dense(self.num_classes, activation="softmax")(x)

        model =  keras.models.Model(inputs=input, outputs=output, name = "EEGModel")
        learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])

        model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss=keras.losses.CategoricalCrossentropy(),
                metrics=[keras.metrics.CategoricalAccuracy()]
        )

        return model

    # def tuneModel(self):
    #     tuner = kt.GridSearch(
    #         hypermodel=self.__tuneModel,
    #         objective="val_categorical_accuracy",
    #         executions_per_trial=1,
    #         overwrite=True,
    #         directory="out",
    #         project_name="eeg_tune",
    #     )

    #     def dataGen():
    #         while True:
    #             for x, y, w in self.db.readChunks(self.batch_size, self.tuner_epochs):
    #                 yield x, y, w

    #     def valGen():
    #         while True:
    #             for x, y, w in self.db.readChunks(self.batch_size, self.tuner_epochs, "validation"):
    #                 yield x, y, w

    #     print(type(dataGen), type(valGen))

    #     tuner.search(
    #         dataGen(),
    #         steps_per_epoch=math.ceil(self.db.sampleNum() / self.batch_size),
    #         epochs=self.tuner_epochs,
    #         validation_data=valGen(),
    #         validation_steps=math.ceil(self.db.sampleNum("validation_tags") / self.batch_size),
    #         # callbacks=[DbRestart()]
    #     )

    #     tuner.results_summary()

    def getValidationElement(self):
        return self.validation_set
    
    def __getValidationData(self):

        print("Downloading validation data.")

        folder = self.validation_set["folder"]
        session = self.validation_set["session"]
        site = self.validation_set["site"]

        channels = ChannSelector().selectEeg(self.aws.loadEegChannelsTsv(folder, session, site))

        parser = self.aws.loadEegEdf(folder, session, site)
        annotations = self.aws.loadEegAnnotationsCsv(folder, session, site)

        parser.setAnottations(annotations)
        features, labels = parser.featuresPerEvent(channels)
        parser.purge()

        self.validation_data = (features, keras.utils.to_categorical(labels, num_classes=self.num_classes))

    def getModel(self):
        return self.model
    
    def __classWeight(self, labels):
        weights = {}
        for label in range(self.num_classes):
            freq = np.count_nonzero(labels == label)

            weights[label] = 1 - (freq/len(labels))

        return weights

    def fit(self, features, labels):

        history = self.model.fit(
            features,
            keras.utils.to_categorical(labels, num_classes=self.num_classes),
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=self.validation_data,
            shuffle=True,
            class_weight=self.__classWeight(labels),
            verbose=0,
            callbacks=[self.callbacks]
        )

        cat_acc = np.mean(history.history['categorical_accuracy'])
        val_cat_acc = np.mean(history.history['val_categorical_accuracy'])
        loss = np.mean(history.history['loss'])
        val_loss = np.mean(history.history['val_loss'])

        return cat_acc, val_cat_acc, loss, val_loss

    def evaluate(self, chunk, tags):
        return self.model.evaluate(tf.stack(chunk), tf.stack(keras.utils.to_categorical(tags, num_classes=5)))
        
    def getMetrics(self):
        return self.model.metrics_names

    def save(self, chunks, mode = "ROWS", done = False):
        self.model.save(self.dir + "mlp_eeg.keras", include_optimizer=True)
        with open(self.dir + "mlp_eeg.chunks", "w") as chunksFile:
            chunksFile.write(f"{mode}={chunks}\n")
            if done == True:
                chunksFile.write(f"DONE")