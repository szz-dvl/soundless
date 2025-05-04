import random
import numpy as np
import pandas as pd
import os
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.utils import shuffle

from modules.aws import AWS
from modules.chann_selector import ChannSelector, MissingChannels
from modules.edf import BadSamplingFreq
from modules.model import EEGModel
from modules.utils import Utils

from dotenv import load_dotenv

load_dotenv()

class TestReserve(Exception):
    pass

aws = AWS()
utils = Utils()
test_reserve = []

CHUNK_SIZE = 2
CHUNKS_PER_TRAIN = 5
CHUNKS_TO_SAVE = 10

# Model save will store the test instances
utils.createDirIfNotExists("out")

def getInfoTask(row: pd.Series):
    folder = row["BidsFolder"]
    session = row["SessionID"]
    site = row["SiteID"]

    channels = ChannSelector().select(aws.loadEegChannelsTsv(folder, session, site))

    if row['HasAnnotations'] == 'Y':
        
        if random.randrange(100) <= 15:
            raise TestReserve()

        parser = aws.loadEegEdf(folder, session, site)
        annotations = aws.loadEegAnnotationsCsv(folder, session, site)
        parser.setAnottations(annotations)
        chunks = parser.crop(channels["name"].to_list())
        tags = parser.getTags()
        parser.purge()

        return chunks, tags
    
    else:
        return None, None
    
def recoverState():
    try:
        with open(os.getenv("MODEL_CHECKPOINT_DIR") + "eeg.chunks", "r") as chunksFile:
            chunksInfo = chunksFile.readline()
            return int(chunksInfo.split("=")[-1])

    except FileNotFoundError:
        return 0

def trainNN():
    global test_reserve

    model = EEGModel()
    getChunk = utils.chunkDataframe(pd.read_csv('bdsp_psg_master_20231101.csv'), CHUNK_SIZE)
    chunksToSkip = recoverState()
    chunks = 0
    Xtrain = []
    ytrain = []


    for chunk in getChunk:
        chunks += 1

        if chunksToSkip < chunks:
            with ThreadPoolExecutor(max_workers=CHUNK_SIZE) as executor:
                futureToRow = {executor.submit(getInfoTask, row): row for _, row in chunk.iterrows()}
                for future in as_completed(futureToRow):
                    row = futureToRow[future]
                    try:
                        
                        data, tags = future.result()
                        
                        if data is not None: 
                            
                            Xtrain.extend(list(map(lambda x: x.get_data(), data)))
                            ytrain.extend(tags)

                            if chunks % CHUNKS_PER_TRAIN == 0:
                                x, y = shuffle(Xtrain, ytrain)
                                accuracy = model.feed(x, y)
                                print(f"\033[1mAccuracy: {accuracy}\033[0m")
                                
                                Xtrain = []
                                ytrain = []

                    except TestReserve as instance:
                        print(f"Reserved for test: {row["BidsFolder"]}, session: {row["SessionID"]}")

                        test_reserve.append({
                            "folder": row["BidsFolder"],
                            "session": row["SessionID"],
                            "site": row["SiteID"]
                        })

                        pass
                    except ClientError:
                        print(f"Missing data for sub: {row["BidsFolder"]}, session: {row["SessionID"]}")
                        pass
                    except MissingChannels as ex:
                        print(f"Missing channels: {row["BidsFolder"]}, session: {row["SessionID"]}", ex)
                        pass
                    except BadSamplingFreq as ex:
                        print(f"Bad sampling frequency ({ex}): {row["BidsFolder"]}, session: {row["SessionID"]}")
                        pass
                    except Exception as exc:
                        print(f"Exception %s: %s"%(row["BidsFolder"], exc))
                        pass
            
            if chunks % CHUNKS_TO_SAVE == 0:
                model.save(chunks, test_reserve)

    model.save(chunks, test_reserve, True)

trainNN()