import os
import pandas as pd

class Utils():
    def __init__(self):
        pass

    def createDirIfNotExists(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path) 

    # https://stackoverflow.com/questions/57198121/select-next-n-rows-in-pandas-dataframe-using-iterrows
    def chunkDataframe(df: pd.DataFrame, chunkSize = 10):
        for startRow in range(0, df.shape[0], chunkSize):
            endRow  = min(startRow + chunkSize, df.shape[0])
            yield df.iloc[startRow:endRow, :]