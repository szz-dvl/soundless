


from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

from modules.utils import Utils
from modules.aws import AWS
from botocore.exceptions import ClientError

class NoPreSleepQuestionary(Exception):
    pass

CHUNK_SIZE = 20

utils = Utils()
aws = AWS()

preSleepKeys = {}
total = 0

def countNotMissing(preSleep: pd.DataFrame):
    global preSleepKeys

    for key in preSleep.loc[~preSleep["value"].isin(["missingData", np.nan]), "key"]:
        try: 
            preSleepKeys[key] += 1
        except KeyError:
            preSleepKeys[key] = 1

    # print(preSleep.loc[preSleep["value"].isin(["missingData", np.nan]), "key"].count())
    # print(preSleep.loc[~preSleep["value"].isin(["missingData", np.nan]), "key"].count())
    # print(preSleep.count())

def getInfoTask(row):
    folder = row["BidsFolder"]
    session = row["SessionID"]
    site = row["SiteID"]

    if row['PreSleepQuestionnaire'] == 'Y':
        return aws.loadEegPreSleepQuestCsv(folder, session, site)
    else:
        raise NoPreSleepQuestionary()

def getPreSleepQuestionary():
    global total

    getChunk = utils.chunkDataframe(pd.read_csv('bdsp_psg_master_20231101.csv'), CHUNK_SIZE)
    # chunks = 0
    for chunk in getChunk:
        # chunks += 1
        # if chunks > 5:
        #     break
        with ThreadPoolExecutor(max_workers=CHUNK_SIZE) as executor:
            futureToRow = {executor.submit(getInfoTask, row): row for _, row in chunk.iterrows()}
            for future in as_completed(futureToRow):
                row = futureToRow[future]
                try:
                    
                    psq = future.result()
                    countNotMissing(psq)
                    total += 1

                except NoPreSleepQuestionary:
                    print(f"Missing pre-sleep questionary for sub: {row["BidsFolder"]}, session: {row["SessionID"]}")
                    pass
                except ClientError:
                    print(f"Missing data for sub: {row["BidsFolder"]}, session: {row["SessionID"]}")
                    pass
                except Exception as exc:
                    print(f"Exception %s: %s"%(row["BidsFolder"], exc))
                    pass

getPreSleepQuestionary()
df = pd.DataFrame.from_dict(preSleepKeys, orient='index', columns=["value"])
df["value"] /= total

print(f"{df.sort_values(by=["value"], ascending=False).to_string()}")
print(df[df["value"] > 0.9].index.to_list())