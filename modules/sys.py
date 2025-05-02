import os

class System():
    def __init__(self):
        pass

    def createDirIfNotExists(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path) 
