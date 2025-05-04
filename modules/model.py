import json
import keras
from keras import layers
from dotenv import load_dotenv
import os
import tensorflow as tf

load_dotenv()

class EEGModel():
    
    def __init__(self, outdir = "out/"):

        self.outdir = outdir
        self.dir = os.getenv("MODEL_CHECKPOINT_DIR")

        if os.path.exists(self.dir + "eeg.keras"):
            self.model = keras.models.load_model(self.dir + "eeg.keras", compile=True)
        else:
            input = keras.Input(shape=(18,6001), name="EGGInput")
            x = layers.Dense(768, activation="relu", name="EGGDense768")(input)
            x = layers.Dense(64, activation="relu", name="EGGDense64")(x)
            x = layers.Dense(32, activation="relu", name="EGGDense32")(x)
            outputs = layers.Dense(1, activation="sigmoid", name="EGGOutput")(x)
            self.model = keras.Model(inputs = input, outputs = outputs, name = "EEGModel")
            self.model.compile(
                optimizer=keras.optimizers.Adam(),
                loss=keras.losses.MeanSquaredError(),
                metrics=[keras.metrics.Accuracy()]
            )

        self.model.summary()

    def getModel(self):
        return self.model

    def feed(self, chunk, tags):
        self.model.fit(tf.stack(chunk), tf.stack(tags))

    def evaluate(self, chunk, tags):
        return self.model.evaluate(tf.stack(chunk), tf.stack(tags))
        
    def getMetrics(self):
        return self.model.metrics_names

    def save(self, chunks, test_instances, done = False):
        self.model.save(self.dir + "eeg.keras", include_optimizer=True)
        with open(self.dir + "eeg.chunks", "w") as chunksFile:
            chunksFile.write(f"chunks = {chunks}\n")
            if done == True:
                chunksFile.write(f"DONE")

        with open(self.outdir + "test_instances.json", "w") as jsonFile:
            jsonFile.writelines(json.dumps(test_instances, indent=4))
