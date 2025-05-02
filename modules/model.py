import keras
from keras import layers
from dotenv import load_dotenv
import os

load_dotenv()

class EEGModel():
    
    def __init__(self):

        self.dir = os.getenv("MODEL_CHECKPOINT_DIR")

        if os.path.exists(self.dir + "eeg.keras"):
            self.model = keras.models.load_model(self.dir + "eeg.keras")
        else:
            input = keras.Input(shape=(18,), name="EGGInput") 
            x = layers.Dense(12, activation="relu", name="EGGDense12")(input)
            x = layers.Dense(6, activation="relu", name="EGGDense6")(x)
            outputs = layers.Dense(1, activation="softmax", name="EGGOutput")(x)
            self.model = keras.Model(inputs = input, outputs = outputs, name = "EEGModel")
            self.model.compile(
                optimizer=keras.optimizers.Adam(),
                loss=keras.losses.MeanSquaredError(),
                metrics=[keras.metrics.Accuracy()]
            )

    def getModel(self):
        return self.model

    def feed(self, chunk, tags):
        self.model.train_on_batch(chunk, tags)

    def save(self, chunks):
        self.model.save(self.dir + "eeg.keras")
        with open(self.dir + "eeg.chunks", "w") as chunksFile:
            chunksFile.write(f"chunks = {chunks}")
