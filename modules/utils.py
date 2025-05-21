from collections import Counter
import os
import pandas as pd
from imblearn.over_sampling import SMOTE

class Utils():
    def __init__(self):
        pass

    def createDirIfNotExists(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path) 

    # https://stackoverflow.com/questions/57198121/select-next-n-rows-in-pandas-dataframe-using-iterrows
    def chunkDataframe(self, df: pd.DataFrame, chunkSize = 10):
        for startRow in range(0, df.shape[0], chunkSize):
            endRow  = min(startRow + chunkSize, df.shape[0])
            yield df.iloc[startRow:endRow, :]

    def oversample(self, features, labels):
        counter = Counter(labels)
        maximum = max(counter, key=counter.get)

        smote = SMOTE(sampling_strategy={
            1: counter[maximum]
        })

        return smote.fit_resample(features, labels)

        

