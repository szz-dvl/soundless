import os
import pandas as pd
import csv
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed

from modules.aws import AWS
from modules.sys import System

aws = AWS()
system = System()

CHUNK_SIZE = 20

class BadChannelCount(Exception):
    pass

# https://stackoverflow.com/questions/57198121/select-next-n-rows-in-pandas-dataframe-using-iterrows
def chunkDataframe(df: pd.DataFrame, chunkSize = 10):
    for startRow in range(0, df.shape[0], chunkSize):
        endRow  = min(startRow + chunkSize, df.shape[0])
        yield df.iloc[startRow:endRow, :]

def getInfoTask(row: pd.Series):
    folder = row["BidsFolder"]
    session = row["SessionID"]
    site = row["SiteID"]

    document = aws.loadEegJson(folder, session, site)
    channels = aws.loadEegChannelsTsv(folder, session, site)

    channs = channels["name"].to_list()
    channs.sort()

    return channs, document["EEGChannelCount"]


def parseMainTask(writer, file):
    getChunk = chunkDataframe(pd.read_csv('bdsp_psg_master_20231101.csv'), CHUNK_SIZE)

    for chunk in getChunk:
        with ThreadPoolExecutor(max_workers=CHUNK_SIZE) as executor:
            futureToRow = {executor.submit(getInfoTask, row): row for _, row in chunk.iterrows()}
            for future in as_completed(futureToRow):
                row = futureToRow[future]
                try:
                    channels, count = future.result()
                    
                    if len(channels) != count:
                        raise BadChannelCount();

                    writer.writerow([row["BidsFolder"], row["SessionID"], count, "|".join(channels)])
                    file.flush()
                            
                except ClientError:
                    print(f"Missing data for sub: {row["BidsFolder"]}, session: {row["SessionID"]}")
                    pass
                except BadChannelCount:
                    print(f"Bad channel count: {row["BidsFolder"]}, session: {row["SessionID"]}")
                    pass
                except Exception as exc:
                    print(f"Exception %s: %s"%(row["BidsFolder"], exc))
                    pass


system.createDirIfNotExists("out")
with open('out/channels.csv','w') as channels:
    writer = csv.writer(channels, delimiter=',', quotechar='"')
    parseMainTask(writer, channels)

    # 27-04-2025 Staging folder structure has changed since last viewed, there are no participants.tsv and bids_staging folder has been deleted on favour of bids folder
    # that is holding all the data for all the ¿sites?. I'm focusing on the patiens listed in "bdsp_psg_master_20231101.csv"
