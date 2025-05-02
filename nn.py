import pandas as pd
import os
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed

from modules.aws import AWS
from modules.chann_selector import ChannSelector, MissingChannels
from modules.model import EEGModel

from dotenv import load_dotenv

load_dotenv()

aws = AWS()

CHUNK_SIZE = 2

# https://stackoverflow.com/questions/57198121/select-next-n-rows-in-pandas-dataframe-using-iterrows
def chunkDataframe(df: pd.DataFrame, chunkSize = 10):
    for startRow in range(0, df.shape[0], chunkSize):
        endRow  = min(startRow + chunkSize, df.shape[0])
        yield df.iloc[startRow:endRow, :]

def getInfoTask(row: pd.Series):
    folder = row["BidsFolder"]
    session = row["SessionID"]
    site = row["SiteID"]

    channels = ChannSelector().select(aws.loadEegChannelsTsv(folder, session, site))

    if row['HasAnnotations'] == 'Y':
    
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
            chunksInfo = chunksFile.read()
            return int(chunksInfo.split("=")[-1])

    except FileNotFoundError:
        return 0

ok = 0
all = 0

def parseMainTask():
    global all, ok

    model = EEGModel()
    getChunk = chunkDataframe(pd.read_csv('bdsp_psg_master_20231101.csv'), CHUNK_SIZE)
    chunksToSkip = recoverState()
    chunks = 0

    for chunk in getChunk:
        chunks += 1

        if chunksToSkip < chunks:
            with ThreadPoolExecutor(max_workers=CHUNK_SIZE) as executor:
                futureToRow = {executor.submit(getInfoTask, row): row for _, row in chunk.iterrows()}
                for future in as_completed(futureToRow):
                    row = futureToRow[future]
                    try:
                        all += 1
                        data, tags = future.result()
                        
                        if data is not None:
                            model.feed(list(map(lambda x: x.get_data(), data)), tags)
                            ok += 1
        
                    except ClientError:
                        print(f"Missing data for sub: {row["BidsFolder"]}, session: {row["SessionID"]}")
                        pass
                    except MissingChannels as ex:
                        print(f"Missing channels: {row["BidsFolder"]}, session: {row["SessionID"]}", ex)
                        pass
                    except Exception as exc:
                        print(f"Exception %s: %s"%(row["BidsFolder"], exc))
                        pass

            model.save(chunks)

parseMainTask()
print(f"Representation: {(ok/all) * 100}%")