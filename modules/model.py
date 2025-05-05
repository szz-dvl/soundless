import json
import keras
from keras import layers
from dotenv import load_dotenv
import os
import tensorflow as tf
from sklearn.utils import gen_batches, shuffle

load_dotenv()

class EEGModel():
    
    def __init__(self, outdir = "out/"):

        tf.get_logger().setLevel('ERROR')

        self.batch_size = 25
        self.outdir = outdir
        self.dir = os.getenv("MODEL_CHECKPOINT_DIR")

        if os.path.exists(self.dir + "eeg.keras"):
            self.model = keras.models.load_model(self.dir + "eeg.keras", compile=True)
        else:
            input = keras.Input(shape=(18,6001), name="EGGInput")
            x = layers.BatchNormalization(name="EGGScaler")(input)
            x = layers.Flatten(name="EEGFlatten")(x)
            x = layers.Dropout(0.2, name="EGGDropout1")(x)
            x = layers.Dense(768, activation="relu", name="EGGDense768")(x)
            x = layers.Dropout(0.2, name="EGGDropout2")(x)
            x = layers.Dense(64, activation="relu", name="EGGDense64")(x)
            outputs = layers.Dense(5, activation="softmax", name="EGGOutput")(x)
            self.model = keras.Model(inputs = input, outputs = outputs, name = "EEGModel")
            self.model.compile(
                optimizer=keras.optimizers.Adam(),
                loss=keras.losses.MeanAbsoluteError(),
                metrics=[keras.metrics.CategoricalAccuracy()]
            )

        self.model.summary()

    def getModel(self):
        return self.model

    def feed(self, chunk, tags) -> float:
        
        result = None
        for slice in gen_batches(len(chunk), self.batch_size):
            x, y = shuffle(chunk[slice], tags[slice])
            result = self.model.train_on_batch(
                tf.stack(x),
                tf.stack(keras.utils.to_categorical(y, num_classes=5)),
                return_dict=True
            )

        return result['categorical_accuracy']

    def evaluate(self, chunk, tags):
        return self.model.evaluate(tf.stack(chunk), tf.stack(keras.utils.to_categorical(tags, num_classes=5)))
        
    def getMetrics(self):
        return self.model.metrics_names

    def save(self, chunks, test_instances, mode = "ROWS", done = False):
        self.model.save(self.dir + "eeg.keras", include_optimizer=True)
        with open(self.dir + "eeg.chunks", "w") as chunksFile:
            chunksFile.write(f"{mode}={chunks}\n")
            if done == True:
                chunksFile.write(f"DONE")

        with open(self.outdir + "test_instances.json", "w") as jsonFile:
            jsonFile.writelines(json.dumps(test_instances, indent=4))
