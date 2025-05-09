import random
import pandas as pd
import os
import sys
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed

from modules.aws import AWS
from modules.chann_selector import ChannSelector, MissingChannels
from modules.edf import BadSamplingFreq
from modules.model import EEGModel
from modules.utils import Utils
from modules.db import Db

from dotenv import load_dotenv

load_dotenv()

class TestReserve(Exception):
    pass

class IncompatibleCheckpoint(Exception):
    pass

class ValidationElement(Exception):
    pass

aws = AWS()
utils = Utils()
db = Db()

CHUNK_SIZE = 3
CHUNKS_TO_SAVE = 10
CHUNKS_PER_TRAIN = 5

validation_set = [
    {
        "folder": "sub-S0001111192396",
        "session": 1,
        "site": "S0001"
    },
    {
        "folder": "sub-S0001111192986",
        "session": 1,
        "site": "S0001"
    },
    {
        "folder": "sub-S0001111201541",
        "session": 1,
        "site": "S0001"
    },
    {
        "folder": "sub-S0001111212020",
        "session": 1,
        "site": "S0001"
    },
    {
        "folder": "sub-S0001111214824",
        "session": 1,
        "site": "S0001"
    }
]

# Model save will store the test instances
utils.createDirIfNotExists("out")

def isValidation(folder, session, site):
    for validation in validation_set:
        if validation["folder"] == folder and validation["session"] == session and validation["site"] == site:
            return True
    
    return False

def parseData(folder, session, site, channels): 
    
    parser = aws.loadEegEdf(folder, session, site)
    annotations = aws.loadEegAnnotationsCsv(folder, session, site)

    parser.setAnottations(annotations)
    chunks = parser.crop(channels["name"].to_list())
    tags = parser.getTags()
    parser.purge()

    return chunks, tags
    
def getInfoTask(row: pd.Series):
    folder = row["BidsFolder"]
    session = row["SessionID"]
    site = row["SiteID"]

    if isValidation(folder, session, site):
        raise ValidationElement()

    channels = ChannSelector().select(aws.loadEegChannelsTsv(folder, session, site))

    if row['HasAnnotations'] == 'Y':
        
        if random.randrange(100) <= 15:
            raise TestReserve()
        
        return parseData(folder, session, site, channels)
    
    else:
        return None, None
    
def recoverState():
    try:
        with open(os.getenv("MODEL_CHECKPOINT_DIR") + "eeg.chunks", "r") as chunksFile:
            chunksInfo = chunksFile.readline()
            parts = chunksInfo.split("=")
            if parts[0] != "CHUNKS":
                raise IncompatibleCheckpoint()
            
            return int(chunksInfo.split("=")[-1])

    except FileNotFoundError:
        return 0

def populateValidation():

    full = db.sampleNum("validation_tags") != 0

    if full:
        print("Skipping validation population.")
        return

    for validation in validation_set:
        try:
            folder = validation["folder"]
            session = validation["session"]
            site = validation["site"]

            channels = ChannSelector().select(aws.loadEegChannelsTsv(folder, session, site))
            chunks, tags = parseData(folder, session, site, channels)
            db.insertChunks(chunks, tags, "validation")

        except Exception as ex:
            print(f"Exception populating validation set %s: %s"%(validation["folder"], ex))
            sys.exit(1)

    print("Populated validation set.")

def trainNN():
    model = EEGModel()
    getChunk = utils.chunkDataframe(pd.read_csv('bdsp_psg_master_20231101.csv'), CHUNK_SIZE)
    chunksToSkip = recoverState()
    chunks = 0
    inserted = 0

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
                            db.insertChunks(data, tags)
                            inserted += 1

                        if inserted == CHUNKS_PER_TRAIN:
                            accuracy, val_accuracy, loss, val_loss = model.fit()
                            print(f"\033[1mAccuracy: {accuracy}, Validation accuracy: {val_accuracy}, Loss: {loss}, Validation loss: {val_loss} \033[0m")

                            inserted = 0
                            db.flushData()

                    except ValidationElement:
                        print(f"Skipped validation element: {row["BidsFolder"]}, session: {row["SessionID"]}")
                        pass                            
                    except TestReserve:
                        print(f"Reserved for test: {row["BidsFolder"]}, session: {row["SessionID"]}")
                        db.insertTest(row["BidsFolder"], row["SessionID"], row["SiteID"])
                        pass
                    except ClientError:
                        print(f"Missing data for sub: {row["BidsFolder"]}, session: {row["SessionID"]}")
                        pass
                    except MissingChannels as ex:
                        print(f"Missing channels: {row["BidsFolder"]}, session: {row["SessionID"]}", ex)
                        pass
                    except BadSamplingFreq:
                        print(f"Bad sampling frequency ({ex}): {row["BidsFolder"]}, session: {row["SessionID"]}")
                        pass
                    except Exception as ex:
                        print(f"Exception %s: %s"%(row["BidsFolder"], ex))
                        pass
            
            if chunks % CHUNKS_TO_SAVE == 0:
                model.save(chunks, "CHUNKS")

    model.save(chunks, "CHUNKS", True)

populateValidation()
trainNN()
db.close()