import random
import sys
import pandas as pd
import os
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed

from modules.aws import AWS
from modules.chann_selector import ChannSelector, MissingChannels
from modules.edf import BadSamplingFreq
from modules.mlp import MLPEegModel
from modules.utils import Utils
from modules.db_mlp import Db

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
CHUNKS_PER_TRAIN = 200

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
    },
    {
        "folder": "sub-S0001111193967",
        "session": 1,
        "site": "S0001"
    },
    {
        "folder": "sub-S0001111198326",
        "session": 1,
        "site": "S0001"
    },
    {
        "folder": "sub-S0001111200447",
        "session": 1,
        "site": "S0001"
    },
    {
        "folder": "sub-S0001111204219",
        "session": 2,
        "site": "S0001"
    },
    {
        "folder": "sub-S0001111229530",
        "session": 1,
        "site": "S0001"
    },
    {
        "folder": "sub-S0001111231120",
        "session": 1,
        "site": "S0001"
    },
    {
        "folder": "sub-S0001111232048",
        "session": 1,
        "site": "S0001"
    },
    {
        "folder": "sub-S0001111238264",
        "session": 1,
        "site": "S0001"
    },
    {
        "folder": "sub-S0001111241357",
        "session": 1,
        "site": "S0001"
    },
    {
        "folder": "sub-S0001111250016",
        "session": 1,
        "site": "S0001"
    },
    {
        "folder": "sub-S0001111256738",
        "session": 1,
        "site": "S0001"
    },
    {
        "folder": "sub-S0001111260624",
        "session": 1,
        "site": "S0001"
    },
    {
        "folder": "sub-S0001111265591",
        "session": 1,
        "site": "S0001"
    },
    {
        "folder": "sub-S0001111266484",
        "session": 2,
        "site": "S0001"
    },
    {
        "folder": "sub-S0001111191757",
        "session": 1,
        "site": "S0001"
    },
]

def isValidation(folder, session, site):
    for validation in validation_set:
        if validation["folder"] == folder and validation["session"] == session and validation["site"] == site:
            return True
    
    return False

def parseData(folder, session, site, channels): 
    
    parser = aws.loadEegEdf(folder, session, site)
    annotations = aws.loadEegAnnotationsCsv(folder, session, site)

    parser.setAnottations(annotations)
    features, labels = parser.featuresPerEvent(channels)
    parser.purge()

    return features, labels
    
def getInfoTask(row: pd.Series):
    folder = row["BidsFolder"]
    session = row["SessionID"]
    site = row["SiteID"]

    if isValidation(folder, session, site):
        raise ValidationElement()

    channels = ChannSelector().selectEeg(aws.loadEegChannelsTsv(folder, session, site))

    if row['HasAnnotations'] == 'Y':

        if random.randrange(100) <= 10:
            raise TestReserve()
                
        return parseData(folder, session, site, channels)
    
    else:
        return None, None

def populateValidation():

    full = db.sampleNum("validation") != 0

    if full:
        print("Skipping validation population.")
        return

    for validation in validation_set:
        try:
            folder = validation["folder"]
            session = validation["session"]
            site = validation["site"]

            channels = ChannSelector().selectEeg(aws.loadEegChannelsTsv(folder, session, site))
            features, labels = parseData(folder, session, site, channels)
            db.insertFeatures(features, labels, "validation")

        except Exception as ex:
            print(f"Exception populating validation set %s: %s"%(validation["folder"], ex))
            sys.exit(1)

    print("Populated validation set.")

def recoverState():
    try:
        with open(os.getenv("MODEL_CHECKPOINT_DIR") + "mlp_eeg.chunks", "r") as chunksFile:
            chunksInfo = chunksFile.readline()
            chunkParts = chunksInfo.split("=")
            if chunkParts[0] != "CHUNKS":
                raise IncompatibleCheckpoint()
            
            insertedInfo = chunksFile.readline()
            insertedParts = insertedInfo.split("=")
            inserted = int(insertedParts[-1])

            if inserted == 0:
                db.flushData()
            
            return int(chunkParts[-1]), inserted

    except FileNotFoundError:
        return 0, 0

def trainMLP():
    model = MLPEegModel()
    getChunk = utils.chunkDataframe(pd.read_csv('bdsp_psg_master_20231101.csv'), CHUNK_SIZE)
    chunksToSkip, inserted = recoverState()
    chunks = 0

    for chunk in getChunk:
        chunks += 1

        if chunksToSkip < chunks:
            with ThreadPoolExecutor(max_workers=CHUNK_SIZE) as executor:
                futureToRow = {executor.submit(getInfoTask, row): row for _, row in chunk.iterrows()}
                for future in as_completed(futureToRow):
                    row = futureToRow[future]
                    try:
                        
                        features, labels = future.result()

                        if features is not None:
                            db.insertFeatures(features, labels)
                            inserted += 1

                        if inserted == CHUNKS_PER_TRAIN:
                            cat_acc, val_cat_acc, loss, val_loss = model.fit()
                            print(f"\033[1mAccuracy: {cat_acc}, Validation accuracy: {val_cat_acc}, Loss: {loss}, Validation loss: {val_loss}\033[0m")
                            model.save(chunks, 0, "CHUNKS")

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
                    except BadSamplingFreq as ex:
                        print(f"Bad sampling frequency ({ex}): {row["BidsFolder"]}, session: {row["SessionID"]}")
                        pass
                    except Exception as ex:
                        print(f"Exception %s: %s"%(row["BidsFolder"], ex))
                        pass
            
            model.save(chunks, inserted, "CHUNKS", False, True)

    model.save(chunks, inserted, "CHUNKS", True)

populateValidation()
trainMLP()
db.close()