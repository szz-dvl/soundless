import os
from datetime import datetime
import pandas as pd
from modules.db_soundless import Db

db = Db()

PREFIX_DATA = "data/soundless-data-no-geo"
PREFIX_USER = "data/soundless-history"
COLUMNS = ['uuid', 'timestamp', 'dB', 'heartRate', 'sleepStage']

def parseFileNameData(filename: str) -> tuple:
    parts = filename.split("_")

    if parts[1] == "province":
        location = "_".join([parts[0], parts[1]])
        date = parts[2]
    else:
        location = parts[0]
        date = parts[1]

    return datetime.strptime(date.split(".")[0], '%Y-%m-%d'), location

for file in os.listdir(PREFIX_DATA):
    try: 
        date, location = parseFileNameData(file)
        df = pd.read_csv(PREFIX_DATA + "/" + file)
        extra = [col for col in df.columns if col not in COLUMNS]

        if len(extra) != 0:
            print(f"Removing extra columns for file: {file}: {extra}")
            df = df.drop(columns=extra)

        db.insertSamples(df, date, location)

    except Exception as ex:
        print(f"Exceition in file {file}: {ex}")

for user in os.listdir(PREFIX_USER):
    try: 
        df = pd.read_csv(PREFIX_USER + "/" + user, header=None, index_col=False).transpose()
        df["user"] = user
        df = df.rename(columns={0: "uuid"})

        db.insertUserSessions(df)

    except Exception as ex:
        print(f"Exceition in user {user}: {ex}")

