import json
import math
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

        self.lr = 0.001
        self.epochs = 5
        self.batch_size = 25
        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        self.outdir = outdir
        self.dir = os.getenv("MODEL_CHECKPOINT_DIR")

        if os.path.exists(self.dir + "eeg.keras"):
            self.model = keras.models.load_model(self.dir + "eeg.keras", compile=True)
        else:
            input = keras.Input(shape=(18,6001), name="EGGInput")
            x = layers.Conv1D(32, (3), strides=(2), use_bias=False, name="EEGConv")(input)
            x = layers.BatchNormalization(scale=False, center=True, name="EGGScaler")(x)
            x = layers.Activation(activation="relu", name="EGGReLU")(x)
            x = layers.MaxPooling1D(pool_size=(2), name="EEGMaxPooling")(x)
            x = layers.Dropout(0.3, name="EGGDropoutPre")(x)
            x = layers.LSTM(50, name="EEGLstm")(x)
            x = layers.Dropout(0.3, name="EGGDropoutPost")(x)
            outputs = layers.Dense(5, activation="softmax", name="EGGOutput")(x)
            self.model = keras.Model(inputs = input, outputs = outputs, name = "EEGModel")
            self.model.compile(
                optimizer=self.optimizer,
                loss=keras.losses.CategoricalCrossentropy(),
                metrics=[keras.metrics.CategoricalAccuracy()]
            )

        self.model.summary()

    def getModel(self):
        return self.model

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
