from concurrent.futures import ThreadPoolExecutor, as_completed
from modules.db_soundless import Db
from modules.utils import Utils
from modules.aws import AWS
import pandas as pd

class NoInfo(Exception):
    pass

class UnableToComputeEffectiveness(Exception):
    pass

class UnableToComputeQuality(Exception):
    pass

utils = Utils()
db = Db()
aws = AWS()

CHUNK_SIZE = 20
SLEEPING = [ "Sleep_stage_N2", "Sleep_stage_2", "Sleep_stage_1", "Sleep_stage_N1", "Sleep_stage_N3", "Sleep_stage_3", "Sleep_stage_REM", "Sleep_stage_R" ]
DEEP_SLEEP = [ "Sleep_stage_N3", "Sleep_stage_3" ]
REM_SLEEP = [ "Sleep_stage_REM", "Sleep_stage_R" ]
LIGHT_SLEEP = [ "Sleep_stage_N2", "Sleep_stage_2", "Sleep_stage_1", "Sleep_stage_N1" ]
NOT_SLEEPING = [ "Sleep_stage_W" ]

data = db.getSleepSficiency()
data.rename(columns={"uuid": "session"})
db.insertAggregatedSoundless(data[data["se"].notnull() & data["se"] > 0])

def computeEffectiveness(annotations: pd.DataFrame):
    time_sleeping = annotations.loc[annotations["event"].isin(SLEEPING), "duration"].sum()
    time_awake = annotations.loc[annotations["event"].isin(NOT_SLEEPING), "duration"].sum()
    total_time = time_sleeping + time_awake

    if total_time == 0:
        raise UnableToComputeEffectiveness()

    return float(time_sleeping/total_time)

def computeQuality(annotations: pd.DataFrame):
    time_sleeping = annotations.loc[annotations["event"].isin(SLEEPING), "duration"].sum()
    time_light = annotations.loc[annotations["event"].isin(LIGHT_SLEEP), "duration"].sum()
    time_deep = annotations.loc[annotations["event"].isin(DEEP_SLEEP), "duration"].sum()
    time_rem = annotations.loc[annotations["event"].isin(REM_SLEEP), "duration"].sum()

    if time_sleeping == 0:
        raise UnableToComputeQuality()

    return float(time_light / time_sleeping), float(time_deep / time_sleeping), float(time_rem / time_sleeping)

def getInfoTask(row):
    folder = row["BidsFolder"]
    session = row["SessionID"]
    site = row["SiteID"]

    if row['HasAnnotations'] == 'Y' and row['PreSleepQuestionnaire'] == 'Y':
        annotations = aws.loadEegAnnotationsCsv(folder, session, site)
        se = computeEffectiveness(annotations)
        light, deep, rem = computeQuality(annotations)
        db.insertAggregatedHSP("-".join([folder, str(session)]), se, light, deep, rem)
        return se
    else:
        raise NoInfo()
    
def getSleepEffectiveness():
    getChunk = utils.chunkDataframe(pd.read_csv('bdsp_psg_master_20231101.csv'), CHUNK_SIZE)

    for chunk in getChunk:
        with ThreadPoolExecutor(max_workers=CHUNK_SIZE) as executor:
            futureToRow = {executor.submit(getInfoTask, row): row for _, row in chunk.iterrows()}
            for future in as_completed(futureToRow):
                row = futureToRow[future]
                try:
                    
                    future.result()

                except UnableToComputeEffectiveness:
                    print(f"Unable to compute effectiveness for sub: {row["BidsFolder"]}, session: {row["SessionID"]}")
                    pass
                except UnableToComputeQuality:
                    print(f"Unable to compute quality for sub: {row["BidsFolder"]}, session: {row["SessionID"]}")
                    pass
                except NoInfo:
                    #print(f"Missing annotations/pre-sleep-q for sub: {row["BidsFolder"]}, session: {row["SessionID"]}")
                    pass
                except Exception as exc:
                    print(f"\033[1mException %s: %s\033[0m"%(row["BidsFolder"], exc))
                    pass

getSleepEffectiveness()

