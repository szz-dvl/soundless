from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
import math
import numpy as np
import pandas as pd
from dateutil import parser
from botocore.exceptions import ClientError

from modules.utils import Utils
from modules.aws import AWS
from modules.db_se import Db

class NoInfo(Exception):
    pass

class MissingSex(Exception):
    pass

class UnableToParseHeight(Exception):
    pass

class BadWeight(Exception):
    pass

class BadFallAsleep(Exception):
    pass

class BadWakeUp(Exception):
    pass

class MissingSmoke(Exception):
    pass

class UnableToComputeEffectiveness(Exception):
    pass

class MissingInterestInResearch(Exception):
    pass

CHUNK_SIZE = 20
SLEEPING = [ "Sleep_stage_N2", "Sleep_stage_2", "Sleep_stage_1", "Sleep_stage_N1", "Sleep_stage_N3", "Sleep_stage_3", "Sleep_stage_REM", "Sleep_stage_R" ]
NOT_SLEEPING = [ "Sleep_stage_W" ]

utils = Utils()
aws = AWS()
db = Db()

def computeEffectiveness(annotations: pd.DataFrame):
    time_sleeping = annotations.loc[annotations["event"].isin(SLEEPING), "duration"].sum()
    time_awake = annotations.loc[annotations["event"].isin(NOT_SLEEPING), "duration"].sum()
    total_time = time_sleeping + time_awake

    if total_time == 0:
        raise UnableToComputeEffectiveness()

    return float((time_sleeping/total_time) * 100)

def getSex(preSleepQ: pd.DataFrame, sub: str) -> int:
    if preSleepQ["sexM"].item() == "1":
        return 1
    elif preSleepQ["sexF"].item() == "1":
        return 2
    
    res = db.getSexForSub(sub)
    if res is not None:
        return res
    
    return 0
    
def getHeight(preSleepQ: pd.DataFrame) -> float:
    raw = str(preSleepQ["height"].item()).lower().replace("’", " ft ").replace("”", " in")
    
    try:
        int(raw)

        if len(raw) == 1:
            return float(raw)
        elif len(raw) == 2:
            return float(raw.split("")[0]) + float(raw.split("")[1]) / 12

        raise UnableToParseHeight(preSleepQ["height"].item())
    
    except ValueError:
        last = 0
        magnitude = np.nan
        for part in raw.split(" "):
            try:
                value = int(part)
                last = value
            except ValueError:
                if part in ["ft", "feet"]:
                    magnitude = last
                elif part in ["in", "inches"]:
                    magnitude += last/12

        if math.isnan(magnitude):
            raise UnableToParseHeight(preSleepQ["height"].item())
    
        return magnitude

def getWeight(preSleepQ: pd.DataFrame) -> float:
    # raw = float(str(preSleepQ["weightInLab"].item()).replace(" ", ""))
    # if not math.isnan(raw) and raw != 0:
    #     return raw
    
    raw = float(str(preSleepQ["weight"].item()).replace(" ", ""))
    if not math.isnan(raw) and raw != 0:
        return raw
    
    raise BadWeight(preSleepQ["weight"].item())

def convertTime(time: str, ini = False):
    raw = None
    toAdd = None

    if time.lower().endswith("am"):
        raw = time.lower().split("am")[0].rstrip(" ")

        hours = int(raw.split(":")[0])
        if hours == 12:
            toAdd = 12
        else:
            toAdd = 0

    elif time.lower().endswith("pm"):
        raw = time.lower().split("pm")[0].rstrip(" ")
        toAdd = 12
    else:
        raw = time.rstrip(" ")

    if toAdd is None:
        if ini:
            hours = int(raw.split(":")[0])
            if hours > 3 and hours <= 12:
                toAdd = 12
            else:
                toAdd = 0
        else:
            toAdd = 0
    
    return parser.parse(f"01/01/1970 {raw}").replace(tzinfo=None) + timedelta(hours = toAdd)

def getHoursInBed(preSleepQ: pd.DataFrame) -> float:
    ini = str(preSleepQ["getIntoBedAt"].item())
    fi = str(preSleepQ["getOutOfBedAt"].item())

    if "-" in ini:
        ini = ini.split("-")[0]

    if "-" in fi:
        fi = fi.split("-")[0]

    inidate = convertTime(ini, True)
    fidate = convertTime(fi, False)
    
    return float((fidate - inidate).seconds / 3600)

    
def getFallAsleep(preSleepQ: pd.DataFrame) -> int:
    if preSleepQ['fallAsleep0_10min'].item() == "1":
        return 1
    elif preSleepQ['fallAsleep10_30min'].item() == "1":
        return 2
    elif preSleepQ['fallAsleep30_60min'].item() == "1":
        return 3
    elif preSleepQ['fallAsleep_more60'].item() == "1":
        return 4
    
    raise BadFallAsleep()
    
def getWakeUp(preSleepQ: pd.DataFrame) -> int:
    if preSleepQ['wakeUp0'].item() == "1":
        return 1
    elif preSleepQ['wakeUp1_3'].item() == "1":
        return 2
    elif preSleepQ['wakeUpMore3'].item() == "1":
        return 3
    
    raise BadWakeUp()

def getLegsFeelFunny(preSleepQ: pd.DataFrame) -> int:
    if preSleepQ['legsFeelFunny'].item() == "1":
        if preSleepQ['legsFeelFunnyWorstIn_night'].item() == "1":
            return 2
        elif preSleepQ['legsFeelFunnyWorstIn_morning'].item() == "1":
            return 3
        elif preSleepQ['legsFeelFunnyWorstIn_sameAlways'].item() == "1":
            return 4
        else:
            return 1
    else:
        return 0
    
def getMoveOrStretchSensation(preSleepQ: pd.DataFrame) -> int:
    if preSleepQ['moveOrStretchSensationGets_better'].item() == "1" \
    or preSleepQ['moveOrStretchSensationGets_worse'].item() == "1" \
    or preSleepQ['moveOrStretchSensationGets_same'].item() == "1":

        if preSleepQ['moveOrStretchSensationGets_better'].item() == "1":
            return 1
        elif preSleepQ['moveOrStretchSensationGets_worse'].item() == "1":
            return 2
        elif preSleepQ['moveOrStretchSensationGets_same'].item() == "1":
            return 3
    else:
        return 0
    
def getSmoke(preSleepQ: pd.DataFrame, sub: str) -> int:
    if preSleepQ["smoke_yes"].item() == "1":
        return 1
    elif preSleepQ["smoke_no"].item() == "1":
        return 2
    
    res = db.getSmokeForSub(sub)
    if res is not None:
        return res
    
    return 0
    
def getHighestEducation(preSleepQ: pd.DataFrame) -> int:
    if preSleepQ['highestEducation_HighSchool'].item() == "1" \
    or preSleepQ['highestEducation_College'].item() == "1" \
    or preSleepQ['highestEducation_MastersOrDoctorate'].item() == "1":

        if preSleepQ['highestEducation_HighSchool'].item() == "1":
            return 1
        elif preSleepQ['highestEducation_College'].item() == "1":
            return 2
        elif preSleepQ['highestEducation_MastersOrDoctorate'].item() == "1":
            return 3
    else:
        return 0

def getCurrently(preSleepQ: pd.DataFrame) -> int:
    if preSleepQ['currently_Unemployed'].item() == "1" \
    or preSleepQ['currently_Student'].item() == "1" \
    or preSleepQ['currently_Employed'].item() == "1" \
    or preSleepQ['currently_Retired'].item() == "1":

        if preSleepQ['currently_Unemployed'].item() == "1":
            return 1
        elif preSleepQ['currently_Student'].item() == "1":
            return 2
        elif preSleepQ['currently_Employed'].item() == "1":
            return 3
        elif preSleepQ['currently_Retired'].item() == "1":
            return 4
    else:
        return 0

def getInterestInResearch(preSleepQ: pd.DataFrame) -> int:
    if preSleepQ["interestInResearchYes"].item() == "1":
        return 1
    elif preSleepQ["interestInResearchNo"].item() == "1":
        return 2
    else: 
        return 0

def buildRow(init: list, preSleep: pd.DataFrame) -> list:
    sub = init[0]
    preSleepQ = preSleep.set_index("key").transpose()
    try:
        init.extend([
            int(preSleepQ["age"].iloc[0]), # NOT CATEGORICAL
            getSex(preSleepQ, sub),
            getHeight(preSleepQ),
            getWeight(preSleepQ),
            int(preSleepQ["ESS"].iloc[0]),
            float(preSleepQ["BMI"].iloc[0]),
            int(preSleepQ["evalForSleepiness"].iloc[0]),
            int(preSleepQ["evalForSnoring"].iloc[0]),
            int(preSleepQ["evalForSleepApnea"].iloc[0]),
            int(preSleepQ["evalForInsomnia"].iloc[0]),
            int(preSleepQ["evalForRestlessLegs"].iloc[0]),
            int(preSleepQ["evalForResearch"].iloc[0]),
            int(preSleepQ["evalForDontHaveProb"].iloc[0]),
            getHoursInBed(preSleepQ),
            getFallAsleep(preSleepQ),
            getWakeUp(preSleepQ),
            int(preSleepQ["troubleFallingAsleep"].iloc[0]),
            int(preSleepQ["troubleStayingAsleep"].iloc[0]),
            int(preSleepQ["iSnore"].iloc[0]),
            int(preSleepQ["iStopBreathing"].iloc[0]),
            int(preSleepQ["iWakeUpGasping"].iloc[0]),
            int(preSleepQ["iWakeWithDryMouthOrSoreThroat"].iloc[0]),
            int(preSleepQ["tiredNoMatterHowMuchSleep"].iloc[0]),
            int(preSleepQ["musclesGetWeakLaughOrAngry"].iloc[0]),
            int(preSleepQ["wakeUpParalyzed"].iloc[0]),
            int(preSleepQ["fallAsleepHallucinations"].iloc[0]),
            int(preSleepQ["shortNapFeelRefreshed"].iloc[0]),
            int(preSleepQ["oftenCross3TimeZones"].iloc[0]),
            int(preSleepQ["workIncludesOvernightShifts"].iloc[0]),
            getLegsFeelFunny(preSleepQ),
            getMoveOrStretchSensation(preSleepQ),
            int(preSleepQ["sittingReading"].iloc[0]),
            int(preSleepQ["watchingTV"].iloc[0]),
            int(preSleepQ["sittingInPublic"].iloc[0]),
            int(preSleepQ["passengerCarOneHour"].iloc[0]),
            int(preSleepQ["lyingDownInAfternoon"].iloc[0]),
            int(preSleepQ["sittingAndTalking"].iloc[0]),
            int(preSleepQ["sittingQuietlyAfterLunch"].iloc[0]),
            int(preSleepQ["stoppedFewMinInTraffic"].iloc[0]),
            int(preSleepQ["howLikelyToFallAsleepTotal"].iloc[0]), # NOT CATEGORICAL
            int(preSleepQ["dxAsthmaExerciseInduced"].iloc[0]),
            int(preSleepQ["dxSeizures"].iloc[0]),
            int(preSleepQ["dxHeadTrauma"].iloc[0]),
            int(preSleepQ["dxCancer"].iloc[0]),
            int(preSleepQ["dxStroke"].iloc[0]),
            int(preSleepQ["dxCongestiveHeartFailure"].iloc[0]),
            int(preSleepQ["dxHighBloodPressure"].iloc[0]),
            int(preSleepQ["dxAnxiety"].iloc[0]),
            int(preSleepQ["dxCOPDOrEmphysema"].iloc[0]),
            int(preSleepQ["dxKidneyDisease"].iloc[0]),
            int(preSleepQ["dxBipolarDisorder"].iloc[0]),
            int(preSleepQ["dxDiabetes"].iloc[0]),
            int(preSleepQ["dxHypothyroid"].iloc[0]),
            int(preSleepQ["dxDepression"].iloc[0]),
            int(preSleepQ["dxFibromyalgia"].iloc[0]),
            int(preSleepQ["dxMemoryProblems"].iloc[0]),
            int(preSleepQ["dxPTSD"].iloc[0]),
            int(preSleepQ["dxHeadaches"].iloc[0]),
            int(preSleepQ["dxPacemaker"].iloc[0]),
            int(preSleepQ["FHxSleepDisorder"].iloc[0]),
            int(preSleepQ["languageOtherThanEnglish"].iloc[0]),
            getSmoke(preSleepQ, sub),
            int(preSleepQ["hxSubstanceAbuseOrAlcoholAbuse"].iloc[0]),
            getHighestEducation(preSleepQ),
            getCurrently(preSleepQ),
            int(preSleepQ["disabled"].iloc[0]),
            int(preSleepQ["haveAllergies"].iloc[0]),
            int(preSleepQ["priorSleepTestYesNo"].iloc[0]),
            getInterestInResearch(preSleepQ)
        ])
    except ValueError:
        raise NoInfo()

    return init

def getInfoTask(row):
    folder = row["BidsFolder"]
    session = row["SessionID"]
    site = row["SiteID"]

    if row['HasAnnotations'] == 'Y' and row['PreSleepQuestionnaire'] == 'Y':
        se = computeEffectiveness(aws.loadEegAnnotationsCsv(folder, session, site))
        row = buildRow([folder, session, site], aws.loadEegPreSleepQuestCsv(folder, session, site))
        row.append(se)
        db.insertRow(row)
        return se
    else:
        raise NoInfo()

def getSleepEffectivenessAndPreSleepQuestionnaire():
    getChunk = utils.chunkDataframe(pd.read_csv('bdsp_psg_master_20231101.csv'), CHUNK_SIZE)

    for chunk in getChunk:
        with ThreadPoolExecutor(max_workers=CHUNK_SIZE) as executor:
            futureToRow = {executor.submit(getInfoTask, row): row for _, row in chunk.iterrows()}
            for future in as_completed(futureToRow):
                row = futureToRow[future]
                try:
                    
                    future.result()

                except MissingInterestInResearch:
                    print(f"Missing interest in research for sub: {row["BidsFolder"]}, session: {row["SessionID"]}")
                    pass
                except UnableToComputeEffectiveness:
                    print(f"Unable to compute effectiveness for sub: {row["BidsFolder"]}, session: {row["SessionID"]}")
                    pass
                except MissingSmoke:
                    print(f"Missing smoke for sub: {row["BidsFolder"]}, session: {row["SessionID"]}")
                    pass
                except BadFallAsleep:
                    print(f"Bad fallAsleep for sub: {row["BidsFolder"]}, session: {row["SessionID"]}")
                    pass
                except BadWeight as ex:
                    print(f"Bad weight for sub: {row["BidsFolder"]}, session: {row["SessionID"]} ({ex})")
                    pass
                except BadWakeUp:
                    print(f"Bad wake up for sub: {row["BidsFolder"]}, session: {row["SessionID"]}")
                    pass
                except UnableToParseHeight as ex:
                    print(f"Unable to parse height for sub: {row["BidsFolder"]}, session: {row["SessionID"]} ({ex})")
                    pass
                except MissingSex:
                    print(f"Missing sex for sub: {row["BidsFolder"]}, session: {row["SessionID"]}")
                    pass
                except NoInfo:
                    #print(f"Missing annotations/pre-sleep-q for sub: {row["BidsFolder"]}, session: {row["SessionID"]}")
                    pass
                except ClientError:
                    #print(f"Missing data for sub: {row["BidsFolder"]}, session: {row["SessionID"]}")
                    pass
                except Exception as exc:
                    print(f"\033[1mException %s: %s\033[0m"%(row["BidsFolder"], exc))
                    pass

getSleepEffectivenessAndPreSleepQuestionnaire()