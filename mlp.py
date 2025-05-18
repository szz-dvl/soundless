import pandas as pd
import os
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed

from modules.aws import AWS
from modules.chann_selector import ChannSelector, MissingChannels
from modules.edf import BadSamplingFreq
from modules.mlp import MLPEegModel
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

def parseData(folder, session, site, channels): 
    
    parser = aws.loadEegEdf(folder, session, site)
    annotations = aws.loadEegAnnotationsCsv(folder, session, site)

    parser.setAnottations(annotations)
    features, labels = parser.featuresPerEvent(channels)
    parser.purge()

    return features, labels
    
def getInfoTask(row: pd.Series, validation: dict):
    folder = row["BidsFolder"]
    session = row["SessionID"]
    site = row["SiteID"]

    if validation["folder"] == folder and validation["session"] == session and validation["site"] == site:
        raise ValidationElement()

    channels = ChannSelector().selectEeg(aws.loadEegChannelsTsv(folder, session, site))

    if row['HasAnnotations'] == 'Y':
                
        return parseData(folder, session, site, channels)
    
    else:
        return None, None
    
def recoverState():
    try:
        with open(os.getenv("MODEL_CHECKPOINT_DIR") + "mlp_eeg.chunks", "r") as chunksFile:
            chunksInfo = chunksFile.readline()
            parts = chunksInfo.split("=")
            if parts[0] != "CHUNKS":
                raise IncompatibleCheckpoint()
            
            return int(chunksInfo.split("=")[-1])

    except FileNotFoundError:
        return 0

def trainMLP():
    model = MLPEegModel()
    getChunk = utils.chunkDataframe(pd.read_csv('bdsp_psg_master_20231101.csv'), CHUNK_SIZE)
    chunksToSkip = recoverState()
    chunks = 0

    for chunk in getChunk:
        chunks += 1

        if chunksToSkip < chunks:
            with ThreadPoolExecutor(max_workers=CHUNK_SIZE) as executor:
                futureToRow = {executor.submit(getInfoTask, row, model.getValidationElement()): row for _, row in chunk.iterrows()}
                for future in as_completed(futureToRow):
                    row = futureToRow[future]
                    try:
                        
                        features, labels = future.result()

                        if features is not None:
                            cat_acc, val_cat_acc, loss, val_loss = model.fit(features, labels)
                            print(f"\033[1mAccuracy: {cat_acc}, Validation accuracy: {val_cat_acc}, Loss: {loss}, Validation loss: {val_loss}\033[0m")

                    except ValidationElement:
                        print(f"Skipped validation element: {row["BidsFolder"]}, session: {row["SessionID"]}")
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
            
            if chunks % CHUNKS_TO_SAVE == 0:
                model.save(chunks, "CHUNKS")

    model.save(chunks, "CHUNKS", True)

trainMLP()
db.close()