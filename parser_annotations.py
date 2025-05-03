import math
import pandas as pd
import json
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed
from modules.aws import AWS
from modules.utils import Utils

aws = AWS()
utils = Utils()

CHUNK_SIZE = 20
annotationsSummary = {}

def aggregateAnnotations(rawAnnotations: pd.DataFrame) -> dict:
    events = rawAnnotations['event'].unique()

    aggregated = {}

    for event in events:
        durations = rawAnnotations.loc[rawAnnotations['event'] == event, 'duration'].unique().tolist()
        aggregated[event] = durations

    return aggregated

def getInfoTask(row: pd.Series):
    folder = row["BidsFolder"]
    session = row["SessionID"]
    site = row["SiteID"]

    if row['HasAnnotations'] == 'Y':
        return aggregateAnnotations(aws.loadEegAnnotationsCsv(folder, session, site))
    else:
        return None
 
def durationsToCounts(durations: list) -> dict:
    counts = {}

    for duration in durations:
        if not math.isnan(duration):
            counts[duration] = 1

    return counts

def mergeAnnotations(aggregatedAnnotations: dict) -> None:
    global annotationsSummary

    if aggregatedAnnotations is None:
        return
    
    for event, durations in aggregatedAnnotations.items():
        if event in annotationsSummary:
            annotationsSummary[event]["count"] += 1

            for duration in durations:
                if not math.isnan(duration):
                    if duration in annotationsSummary[event]["durations"]:
                        annotationsSummary[event]["durations"][duration] += 1
                    else:
                        annotationsSummary[event]["durations"][duration] = 1

        else:
            annotationsSummary[event] = {
                "count": 1,
                "durations": durationsToCounts(durations)
            }

def parseMainTask():
    getChunk = utils.chunkDataframe(pd.read_csv('bdsp_psg_master_20231101.csv'), CHUNK_SIZE)

    for chunk in getChunk:
        with ThreadPoolExecutor(max_workers=CHUNK_SIZE) as executor:
            futureToRow = {executor.submit(getInfoTask, row): row for _, row in chunk.iterrows()}
            for future in as_completed(futureToRow):
                row = futureToRow[future]
                try:
                    aggregatedAnnotations = future.result()
                    mergeAnnotations(aggregatedAnnotations)

                except ClientError:
                    print(f"Missing data for sub: {row["BidsFolder"]}, session: {row["SessionID"]}")
                    pass
                except Exception as exc:
                    print(f"Exception %s: %s"%(row["BidsFolder"], exc))
                    pass


parseMainTask()

utils.createDirIfNotExists("out")
with open("out/annotations.json", "w") as outfile:
    outfile.write(json.dumps(annotationsSummary, indent=4))

    # 27-04-2025 Staging folder structure has changed since last viewed, there are no participants.tsv and bids_staging folder has been deleted on favour of bids folder
    # that is holding all the data for all the ¿sites?. I'm focusing on the patiens listed in "bdsp_psg_master_20231101.csv"
