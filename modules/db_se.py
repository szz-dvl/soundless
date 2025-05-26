import pandas as pd
import psycopg2
import os

from dotenv import load_dotenv

load_dotenv()

class Db():

    def __init__(self):

        self.db = os.getenv("DB_SE_NAME")
        self.host = os.getenv("DB_SE_HOST")
        self.user = os.getenv("DB_SE_USER")
        self.pwd = os.getenv("DB_SE_PASS")
        
        self.conn = psycopg2.connect(
            host=self.host, 
            database=self.db,
            user=self.user, 
            password=self.pwd
        )

        with self.conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS public.se (
                    folder varchar NOT NULL,
                    ses varchar NOT NULL,
                    site varchar NOT NULL,
                    age int NOT NULL,
                    sex smallint NOT NULL,
                    height float8 NOT NULL,
                    weight float8 NOT NULL,
                    ess int NOT NULL,
                    bmi float8 NOT NULL,
                    evalForSleepiness smallint NOT NULL,
                    evalForSnoring smallint NOT NULL,
                    evalForSleepApnea smallint NOT NULL,
                    evalForInsomnia smallint NOT NULL,
                    evalForRestlessLegs smallint NOT NULL,
                    evalForResearch smallint NOT NULL, 
                    evalForDontHaveProb smallint NOT NULL,
                    hoursInBed float8 NOT NULL,
                    fallAsleep smallint NOT NULL,
                    wakeUp smallint NOT NULL,  
                    troubleFallingAsleep smallint NOT NULL, 
                    troubleStayingAsleep smallint NOT NULL, 
                    iSnore smallint NOT NULL,
                    iStopBreathing smallint NOT NULL, 
                    iWakeUpGasping smallint NOT NULL, 
                    iWakeWithDryMouthOrSoreThroat smallint NOT NULL, 
                    tiredNoMatterHowMuchSleep smallint NOT NULL, 
                    musclesGetWeakLaughOrAngry smallint NOT NULL, 
                    wakeUpParalyzed smallint NOT NULL,
                    fallAsleepHallucinations smallint NOT NULL, 
                    shortNapFeelRefreshed smallint NOT NULL, 
                    oftenCross3TimeZones smallint NOT NULL, 
                    workIncludesOvernightShifts smallint NOT NULL,
                    legsFeelFunny smallint NOT NULL, 
                    moveOrStretchSensation smallint NOT NULL, 
                    sittingReading smallint NOT NULL, 
                    watchingTV smallint NOT NULL, 
                    sittingInPublic smallint NOT NULL, 
                    passengerCarOneHour smallint NOT NULL, 
                    lyingDownInAfternoon smallint NOT NULL, 
                    sittingAndTalking smallint NOT NULL, 
                    sittingQuietlyAfterLunch smallint NOT NULL, 
                    stoppedFewMinInTraffic smallint NOT NULL, 
                    howLikelyToFallAsleepTotal int NOT NULL, 
                    dxAsthmaExerciseInduced smallint NOT NULL, 
                    dxSeizures smallint NOT NULL, 
                    dxHeadTrauma smallint NOT NULL, 
                    dxCancer smallint NOT NULL, 
                    dxStroke smallint NOT NULL, 
                    dxCongestiveHeartFailure smallint NOT NULL, 
                    dxHighBloodPressure smallint NOT NULL, 
                    dxAnxiety smallint NOT NULL, 
                    dxCOPDOrEmphysema smallint NOT NULL, 
                    dxKidneyDisease smallint NOT NULL, 
                    dxBipolarDisorder smallint NOT NULL, 
                    dxDiabetes smallint NOT NULL, 
                    dxHypothyroid smallint NOT NULL, 
                    dxDepression smallint NOT NULL, 
                    dxFibromyalgia smallint NOT NULL, 
                    dxMemoryProblems smallint NOT NULL, 
                    dxPTSD smallint NOT NULL, 
                    dxHeadaches smallint NOT NULL, 
                    dxPacemaker smallint NOT NULL, 
                    FHxSleepDisorder smallint NOT NULL, 
                    languageOtherThanEnglish smallint NOT NULL, 
                    smoke smallint NOT NULL, 
                    hxSubstanceAbuseOrAlcoholAbuse smallint NOT NULL, 
                    highestEducation smallint NOT NULL,
                    currently smallint NOT NULL, 
                    disabled smallint NOT NULL, 
                    haveAllergies smallint NOT NULL, 
                    priorSleepTestYesNo smallint NOT NULL, 
                    interestInResearch smallint NOT NULL, 
                    se float8 NOT NULL,
                    CONSTRAINT test_pk PRIMARY KEY (folder,ses)
                );
            """)

        self.conn.commit()

    def reconnect(self):
        self.conn.close()
        self.conn = psycopg2.connect(
            host=self.host, 
            database=self.db,
            user=self.user, 
            password=self.pwd
        )
    
    def insertRow(self, row):
        with self.conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO se (
                    folder,
                    ses,
                    site,
                    age,
                    sex,
                    height,
                    weight,
                    ess,
                    bmi,
                    evalForSleepiness,
                    evalForSnoring,
                    evalForSleepApnea,
                    evalForInsomnia,
                    evalForRestlessLegs,
                    evalForResearch,
                    evalForDontHaveProb,
                    hoursInBed,
                    fallAsleep,
                    wakeUp,
                    troubleFallingAsleep,
                    troubleStayingAsleep,
                    iSnore,
                    iStopBreathing,
                    iWakeUpGasping,
                    iWakeWithDryMouthOrSoreThroat,
                    tiredNoMatterHowMuchSleep,
                    musclesGetWeakLaughOrAngry,
                    wakeUpParalyzed,
                    fallAsleepHallucinations,
                    shortNapFeelRefreshed,
                    oftenCross3TimeZones,
                    workIncludesOvernightShifts,
                    legsFeelFunny,
                    moveOrStretchSensation,
                    sittingReading,
                    watchingTV,
                    sittingInPublic,
                    passengerCarOneHour,
                    lyingDownInAfternoon,
                    sittingAndTalking,
                    sittingQuietlyAfterLunch,
                    stoppedFewMinInTraffic,
                    howLikelyToFallAsleepTotal,
                    dxAsthmaExerciseInduced,
                    dxSeizures,
                    dxHeadTrauma,
                    dxCancer,
                    dxStroke,
                    dxCongestiveHeartFailure,
                    dxHighBloodPressure,
                    dxAnxiety,
                    dxCOPDOrEmphysema,
                    dxKidneyDisease,
                    dxBipolarDisorder,
                    dxDiabetes,
                    dxHypothyroid,
                    dxDepression,
                    dxFibromyalgia,
                    dxMemoryProblems,
                    dxPTSD,
                    dxHeadaches,
                    dxPacemaker,
                    FHxSleepDisorder,
                    languageOtherThanEnglish,
                    smoke,
                    hxSubstanceAbuseOrAlcoholAbuse,
                    highestEducation,
                    currently,
                    disabled,
                    haveAllergies,
                    priorSleepTestYesNo,
                    interestInResearch,
                    se
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s,%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """, 
            row)
        self.conn.commit()
    
    def getSexForSub(self, sub):
        with self.conn.cursor() as cursor:
            cursor.execute(f"SELECT sex FROM se WHERE folder = %s", [sub])
            result = cursor.fetchall()
            if result is None:
                return result
            
            return result[0][0] if len(result) > 0 else None
        
    def getSmokeForSub(self, sub):
        with self.conn.cursor() as cursor:
            cursor.execute(f"SELECT smoke FROM se WHERE folder = %s", [sub])
            result = cursor.fetchall()
            if result is None:
                return result
            
            return result[0][0] if len(result) > 0 else None
    
    def toDataFrame(self):
        with self.conn.cursor() as cursor:
            cursor.execute("""SELECT                     
                                age,
                                sex,
                                height,
                                weight,
                                ess,
                                bmi,
                                evalForSleepiness,
                                evalForSnoring,
                                evalForSleepApnea,
                                evalForInsomnia,
                                evalForRestlessLegs,
                                evalForResearch,
                                evalForDontHaveProb,
                                hoursInBed,
                                fallAsleep,
                                wakeUp,
                                troubleFallingAsleep,
                                troubleStayingAsleep,
                                iSnore,
                                iStopBreathing,
                                iWakeUpGasping,
                                iWakeWithDryMouthOrSoreThroat,
                                tiredNoMatterHowMuchSleep,
                                musclesGetWeakLaughOrAngry,
                                wakeUpParalyzed,
                                fallAsleepHallucinations,
                                shortNapFeelRefreshed,
                                oftenCross3TimeZones,
                                workIncludesOvernightShifts,
                                legsFeelFunny,
                                moveOrStretchSensation,
                                sittingReading,
                                watchingTV,
                                sittingInPublic,
                                passengerCarOneHour,
                                lyingDownInAfternoon,
                                sittingAndTalking,
                                sittingQuietlyAfterLunch,
                                stoppedFewMinInTraffic,
                                howLikelyToFallAsleepTotal,
                                dxAsthmaExerciseInduced,
                                dxSeizures,
                                dxHeadTrauma,
                                dxCancer,
                                dxStroke,
                                dxCongestiveHeartFailure,
                                dxHighBloodPressure,
                                dxAnxiety,
                                dxCOPDOrEmphysema,
                                dxKidneyDisease,
                                dxBipolarDisorder,
                                dxDiabetes,
                                dxHypothyroid,
                                dxDepression,
                                dxFibromyalgia,
                                dxMemoryProblems,
                                dxPTSD,
                                dxHeadaches,
                                dxPacemaker,
                                FHxSleepDisorder,
                                languageOtherThanEnglish,
                                smoke,
                                hxSubstanceAbuseOrAlcoholAbuse,
                                highestEducation,
                                currently,
                                disabled,
                                haveAllergies,
                                priorSleepTestYesNo,
                                interestInResearch,
                                se
                           FROM se;""")
            
            return pd.DataFrame(cursor.fetchall(), columns=[
                        "age",
                        "sex",
                        "height",
                        "weight",
                        "ess",
                        "bmi",
                        "evalForSleepiness",
                        "evalForSnoring",
                        "evalForSleepApnea",
                        "evalForInsomnia",
                        "evalForRestlessLegs",
                        "evalForResearch",
                        "evalForDontHaveProb",
                        "hoursInBed",
                        "fallAsleep",
                        "wakeUp",
                        "troubleFallingAsleep",
                        "troubleStayingAsleep",
                        "iSnore",
                        "iStopBreathing",
                        "iWakeUpGasping",
                        "iWakeWithDryMouthOrSoreThroat",
                        "tiredNoMatterHowMuchSleep",
                        "musclesGetWeakLaughOrAngry",
                        "wakeUpParalyzed",
                        "fallAsleepHallucinations",
                        "shortNapFeelRefreshed",
                        "oftenCross3TimeZones",
                        "workIncludesOvernightShifts",
                        "legsFeelFunny",
                        "moveOrStretchSensation",
                        "sittingReading",
                        "watchingTV",
                        "sittingInPublic",
                        "passengerCarOneHour",
                        "lyingDownInAfternoon",
                        "sittingAndTalking",
                        "sittingQuietlyAfterLunch",
                        "stoppedFewMinInTraffic",
                        "howLikelyToFallAsleepTotal",
                        "dxAsthmaExerciseInduced",
                        "dxSeizures",
                        "dxHeadTrauma",
                        "dxCancer",
                        "dxStroke",
                        "dxCongestiveHeartFailure",
                        "dxHighBloodPressure",
                        "dxAnxiety",
                        "dxCOPDOrEmphysema",
                        "dxKidneyDisease",
                        "dxBipolarDisorder",
                        "dxDiabetes",
                        "dxHypothyroid",
                        "dxDepression",
                        "dxFibromyalgia",
                        "dxMemoryProblems",
                        "dxPTSD",
                        "dxHeadaches",
                        "dxPacemaker",
                        "FHxSleepDisorder",
                        "languageOtherThanEnglish",
                        "smoke",
                        "hxSubstanceAbuseOrAlcoholAbuse",
                        "highestEducation",
                        "currently",
                        "disabled",
                        "haveAllergies",
                        "priorSleepTestYesNo",
                        "interestInResearch",
                        "se"
            ])

    def close(self):
        self.conn.close()
                    

