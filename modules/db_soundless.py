from io import StringIO
import math
import psycopg2
import os
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

class Db():

    def __init__(self):

        self.db = os.getenv("DB_SOUNDLESS_NAME")
        self.host = os.getenv("DB_SOUNDLESS_HOST")
        self.user = os.getenv("DB_SOUNDLESS_USER")
        self.pwd = os.getenv("DB_SOUNDLESS_PASS")
        
        self.conn = psycopg2.connect(
            host=self.host, 
            database=self.db,
            user=self.user, 
            password=self.pwd
        )

        with self.conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS public.samples (
                    uuid varchar NOT NULL,
                    ts int8 NOT NULL,
                    dB int4 NOT NULL,
                    hr int4 NOT NULL,
                    ss int4 NOT NULL,
                    date timestamp NOT NULL,
                    location varchar NOT NULL    
                );
                           
                CREATE TABLE IF NOT EXISTS public.usersession (
                    "user" varchar NOT NULL,
                    uuid varchar NOT NULL  
                );
                           
                CREATE TABLE IF NOT EXISTS public.efficiency (
                    dB float4 NOT NULL,
                    uuid varchar NOT NULL,
                    se float4 NOT NULL  
                );
                           
                CREATE TABLE IF NOT EXISTS public.aggregated (
                    dB float4 NOT NULL,
                    session varchar NOT NULL,
                    se float4 NOT NULL,
                    light float4 NOT NULL,
                    deep float4 NOT NULL,
                    rem float4 NOT NULL,
                    dataset varchar NOT NULL
                );
            """)

        self.conn.commit()


    def insertUserSessions(self, sessions: pd.DataFrame):
        sio = StringIO()
        sessions.to_csv(sio, index=None, header=None)
        sio.seek(0)
        
        with self.conn.cursor() as cursor:           
            cursor.copy_expert(
                sql = f"""
                    COPY usersession (
                        uuid,
                        "user"
                    ) FROM STDIN WITH CSV
                """,
                file=sio
            )
            
        self.conn.commit()

    def insertSamples(self, samples: pd.DataFrame, date: datetime, location: str):
        samples["date"] = date
        samples["location"] = location

        sio = StringIO()
        samples.to_csv(sio, index=None, header=None)
        sio.seek(0)
        
        with self.conn.cursor() as cursor:           
            cursor.copy_expert(
                sql = f"""
                    COPY samples (
                        uuid,  
                        ts,  
                        dB, 
                        hr,
                        ss,
                        date,
                        location
                    ) FROM STDIN WITH CSV
                """,
                file=sio
            )

        self.conn.commit()

    def insertEfficiencies(self, efficiencies: pd.DataFrame):

        sio = StringIO()
        efficiencies.to_csv(sio, index=None, header=None)
        sio.seek(0)
        
        with self.conn.cursor() as cursor:           
            cursor.copy_expert(
                sql = f"""
                    COPY efficiency (
                        dB,
                        uuid,
                        se
                    ) FROM STDIN WITH CSV
                """,
                file=sio
            )
            
        self.conn.commit()

    def insertAggregatedSoundless(self, efficiencies: pd.DataFrame):

        efficiencies["dataset"] = "soundless"
        
        sio = StringIO()
        efficiencies.to_csv(sio, index=None, header=None)
        sio.seek(0)
        
        with self.conn.cursor() as cursor:           
            cursor.copy_expert(
                sql = f"""
                    COPY aggregated (
                        dB,
                        session,
                        se,
                        light,
                        deep,
                        rem,
                        dataset
                    ) FROM STDIN WITH CSV
                """,
                file=sio
            )
            
        self.conn.commit()

    def insertAggregatedHSP(self, session: str, se: float, light: float, deep: float, rem: float):
        
        with self.conn.cursor() as cursor:           
            cursor.execute(
                """
                    INSERT INTO aggregated (
                        dB,
                        session,
                        se,
                        light,
                        deep,
                        rem, 
                        dataset
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (math.nan, session, se, light, deep, rem, "hsp")
            )
            
        self.conn.commit()
    
    def _computeSleepEfficiency(self, row: pd.Series):
        with self.conn.cursor() as cursor:
            cursor.execute("""
                SELECT ts, ss FROM public.samples WHERE uuid = %s ORDER BY ts ASC
            """, (row["uuid"],))
            

            data = pd.DataFrame(cursor.fetchall(), columns=["ts", "ss"])

            last = {
                "ts": None,
                "ss": None
            }

            times = {
                "awake": 0,
                "sleep": 0,
                "light": 0,
                "deep": 0,
                "rem": 0
            }
            
            for _, r in data.iterrows():
                if not last["ts"]:
                    last["ts"] = r["ts"]
                    last["ss"] = r["ss"]
                    
                    continue

                seconds = (r["ts"] - last["ts"]) / 1000
            
                if last["ss"] > 0:
                    times["sleep"] += seconds

                    if last["ss"] == 1:
                        times["light"] += seconds
                    elif last["ss"] == 2:
                        times["deep"] += seconds
                    elif last["ss"] == 3:
                        times["rem"] += seconds 

                else:
                    times["awake"] += seconds

                last["ts"] = r["ts"]
                last["ss"] = r["ss"]

            total = times["awake"] + times["sleep"]
            se = times["sleep"] / total if total > 0 else None
            light = times["light"] / times["sleep"] if times["sleep"] > 0 else None
            deep = times["deep"] / times["sleep"] if times["sleep"] > 0 else None
            rem = times["rem"] / times["sleep"] if times["sleep"] > 0 else None

            return se, light, deep, rem


    def getSleepSficiency(self):
        with self.conn.cursor() as cursor:
            cursor.execute("""
                SELECT avg(dB), uuid 
                FROM public.samples
                group by uuid;
            """)

            data = pd.DataFrame(cursor.fetchall(), columns=["dB", "uuid"])
            
            ses = []
            lights = []
            deeps = []
            rems = []
            for _, row in data.iterrows():
                se, light, deep, rem = self._computeSleepEfficiency(row)
                ses.append(se)
                lights.append(light)
                deeps.append(deep)
                rems.append(rem)

            data["se"] = ses
            data["light"] = lights
            data["deep"] = deeps
            data["rem"] = rems
            
            return data
        
    def getCluster(self):
        with self.conn.cursor() as cursor:
            cursor.execute("""
                SELECT dB, session, se, light, deep, rem, dataset 
                FROM public.aggregated;
            """)

            return pd.DataFrame(cursor.fetchall(), columns=["dB", "session", "se", "light", "deep", "rem", "dataset"])

