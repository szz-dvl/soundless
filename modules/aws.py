from io import BytesIO
import json
import pandas as pd
import boto3 as aws
import os
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from modules.edf import EdfParser

load_dotenv()

class AWS():
    def __init__(self):
        self.bucket = os.getenv("AWS_BUCKET")
        self.region_name = os.getenv("AWS_REGION")
        self.aws_access_key_id = os.getenv("AWS_KEY")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_KEY")
        self.client = None

    def __getAwsCli(self): 

        if self.client is None: 
            session = aws.Session(
                region_name = self.region_name,
                aws_access_key_id = self.aws_access_key_id,
                aws_secret_access_key = self.aws_secret_access_key
            )

            self.client = session.client('s3')

        return self.client

    def __loadAwsFile(self, path: str) -> BytesIO:
        s3 = self.__getAwsCli()
        file = BytesIO()
        s3.download_fileobj(self.bucket, path, file)
        file.seek(0)

        return file

    def __listAwsFolder(self, folder: str) -> list:
        s3 = self.__getAwsCli()
        result = s3.list_objects_v2(Bucket=self.bucket, Prefix=folder, Delimiter="/")

        try:
            return [item["Prefix"] for item in result["CommonPrefixes"]]
        except KeyError:
            return []
        
    def __listAwsFiles(self, folder: str) -> list:
        s3 = self.__getAwsCli()
        result = s3.list_objects_v2(Bucket=self.bucket, Prefix=folder, Delimiter="/")

        try:
            return [item["Key"] for item in result["Contents"]]
        except KeyError:
            return []
        
    def __buildSubPrefix(self, sub: str, session: str, site: str) -> str:
        return f"PSG/bids/{site}/{sub}/ses-{session}/eeg/"

    def __buildJsonPath(self, sub: str, session: str, site: str) -> str:
        return f"{self.__buildSubPrefix(sub,session,site)}{sub}_ses-{session}_task-psg_eeg.json"

    def __buildChannelsTsv(self, sub: str, session: str, site: str) -> str:
        return f"{self.__buildSubPrefix(sub,session,site)}{sub}_ses-{session}_task-psg_channels.tsv"

    def __buildAnnotationsCsv(self, sub: str, session: str, site: str) -> str:
        return f"{self.__buildSubPrefix(sub,session,site)}{sub}_ses-{session}_task-psg_annotations.csv"

    def __buildAnnotationsXltekCsv(self, sub: str, session: str, site: str) -> str:
        return f"{self.__buildSubPrefix(sub,session,site)}{sub}_ses-{session}_task_Xltek.csv"

    def __buildPreSleepQuestCsv(self, sub: str, session: str, site: str) -> str:
        return f"{self.__buildSubPrefix(sub,session,site)}{sub}_ses-{session}_task-psg_pre.csv"

    def __buildPreSleepQuestNoInfoCsv(self, sub: str, session: str, site: str) -> str:
        return f"{self.__buildSubPrefix(sub,session,site)}questionnaire_pre.csv"

    def __buildEdfFile(self, sub: str, session: str, site: str) -> str:
        return f"{self.__buildSubPrefix(sub,session,site)}{sub}_ses-{session}_task-psg_eeg.edf"

    def loadEegJson(self, sub, session, site) -> dict:
        file = self.__loadAwsFile(self.__buildJsonPath(sub, session, site))
        return json.loads(file.getvalue())

    def loadEegChannelsTsv(self, sub, session, site) -> pd.DataFrame:
        file = self.__loadAwsFile(self.__buildChannelsTsv(sub, session, site))
        return pd.read_csv(file, delimiter="\t")

    def loadEegAnnotationsCsv(self, sub, session, site) -> pd.DataFrame:
        try:
            file = self.__loadAwsFile(self.__buildAnnotationsCsv(sub, session, site))
        except ClientError:
            file = self.__loadAwsFile(self.__buildAnnotationsXltekCsv(sub, session, site))

        return pd.read_csv(file)

    def loadEegPreSleepQuestCsv(self, sub, session, site) -> pd.DataFrame:
        try:
            file = self.__loadAwsFile(self.__buildPreSleepQuestCsv(sub, session, site))
        except ClientError:
            file = self.__loadAwsFile(self.__buildPreSleepQuestNoInfoCsv(sub, session, site))

        return pd.read_csv(file, names=["key", "value"], header=0)

    def loadEegEdf(self, sub, session, site) -> EdfParser:
        file = self.__loadAwsFile(self.__buildEdfFile(sub, session, site))
        return EdfParser(file)