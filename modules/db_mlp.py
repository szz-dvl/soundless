import csv
from io import StringIO
import keras
import numpy as np
import pandas as pd
import psycopg2
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.utils import gen_batches, shuffle
import tensorflow as tf

class Db():

    def __init__(self):

        self.num_classes = 5

        self.db = os.getenv("DB_MLP_NAME")
        self.host = os.getenv("DB_MLP_HOST")
        self.user = os.getenv("DB_MLP_USER")
        self.pwd = os.getenv("DB_MLP_PASS")
        
        self.conn = psycopg2.connect(
            host=self.host, 
            database=self.db,
            user=self.user, 
            password=self.pwd
        )

        with self.conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS public.samples (
                    id SERIAL PRIMARY KEY,
                    a_delta float8 NOT NULL,
                    a_theta float8 NOT NULL,
                    a_alpha float8 NOT NULL,
                    a_sigma float8 NOT NULL,
                    a_beta float8 NOT NULL,
                    b_delta float8 NOT NULL,
                    b_theta float8 NOT NULL,
                    b_alpha float8 NOT NULL,
                    b_sigma float8 NOT NULL,
                    b_beta float8 NOT NULL,
                    c_delta float8 NOT NULL,
                    c_theta float8 NOT NULL,
                    c_alpha float8 NOT NULL,
                    c_sigma float8 NOT NULL,
                    c_beta float8 NOT NULL,
                    d_delta float8 NOT NULL,
                    d_theta float8 NOT NULL,
                    d_alpha float8 NOT NULL,
                    d_sigma float8 NOT NULL,
                    d_beta float8 NOT NULL,
                    e_delta float8 NOT NULL,
                    e_theta float8 NOT NULL,
                    e_alpha float8 NOT NULL,
                    e_sigma float8 NOT NULL,
                    e_beta float8 NOT NULL,
                    f_delta float8 NOT NULL,
                    f_theta float8 NOT NULL,
                    f_alpha float8 NOT NULL,
                    f_sigma float8 NOT NULL,
                    f_beta float8 NOT NULL,
                    g_delta float8 NOT NULL,
                    g_theta float8 NOT NULL,
                    g_alpha float8 NOT NULL,
                    g_sigma float8 NOT NULL,
                    g_beta float8 NOT NULL,
                    "label" int4 NOT NULL    
                );

                CREATE TABLE IF NOT EXISTS public.test (
                    folder varchar NOT NULL,
                    ses varchar NOT NULL,
                    site varchar NOT NULL,
                    CONSTRAINT test_pk PRIMARY KEY (folder,ses)
                );
                                       
                CREATE TABLE IF NOT EXISTS public.validation (
                    id SERIAL PRIMARY KEY,
                    a_delta float8 NOT NULL,
                    a_theta float8 NOT NULL,
                    a_alpha float8 NOT NULL,
                    a_sigma float8 NOT NULL,
                    a_beta float8 NOT NULL,
                    b_delta float8 NOT NULL,
                    b_theta float8 NOT NULL,
                    b_alpha float8 NOT NULL,
                    b_sigma float8 NOT NULL,
                    b_beta float8 NOT NULL,
                    c_delta float8 NOT NULL,
                    c_theta float8 NOT NULL,
                    c_alpha float8 NOT NULL,
                    c_sigma float8 NOT NULL,
                    c_beta float8 NOT NULL,
                    d_delta float8 NOT NULL,
                    d_theta float8 NOT NULL,
                    d_alpha float8 NOT NULL,
                    d_sigma float8 NOT NULL,
                    d_beta float8 NOT NULL,
                    e_delta float8 NOT NULL,
                    e_theta float8 NOT NULL,
                    e_alpha float8 NOT NULL,
                    e_sigma float8 NOT NULL,
                    e_beta float8 NOT NULL,
                    f_delta float8 NOT NULL,
                    f_theta float8 NOT NULL,
                    f_alpha float8 NOT NULL,
                    f_sigma float8 NOT NULL,
                    f_beta float8 NOT NULL,
                    g_delta float8 NOT NULL,
                    g_theta float8 NOT NULL,
                    g_alpha float8 NOT NULL,
                    g_sigma float8 NOT NULL,
                    g_beta float8 NOT NULL,
                    "label" int4 NOT NULL    
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

    def flushData(self):
        with self.conn.cursor() as cursor:
            cursor.execute("TRUNCATE TABLE samples RESTART IDENTITY;")
        self.conn.commit()

    def restart(self):
        with self.conn.cursor() as cursor:
            cursor.execute("TRUNCATE TABLE samples RESTART IDENTITY;")
            cursor.execute("TRUNCATE TABLE test;")
        self.conn.commit()
    
    def insertTest(self, folder, session, site):
        with self.conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO test (folder, ses, site)
                VALUES (%s, %s, %s);
            """, 
            [folder, session, site])
        self.conn.commit()

    def getTest(self):
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT * FROM test;")
            return cursor.fetchall()

    # https://medium.com/@askintamanli/fastest-methods-to-bulk-insert-a-pandas-dataframe-into-postgresql-2aa2ab6d2b24
    def insertFeatures(self, features, labels, mode = "samples"):
        rows = np.insert(features, features.shape[-1], labels, axis=1).astype(object)
        rows[:, [rows.shape[-1]-1]] = rows[:, [rows.shape[-1]-1]].astype(int)

        sio = StringIO()
        writer = csv.writer(sio)
        writer.writerows(rows)
        sio.seek(0)

        with self.conn.cursor() as cursor:
            cursor.copy_expert(
                sql = f"""
                    COPY {mode} (
                    a_delta,
                    a_theta,
                    a_alpha,
                    a_sigma,
                    a_beta,
                    b_delta,
                    b_theta,
                    b_alpha,
                    b_sigma,
                    b_beta,
                    c_delta,
                    c_theta,
                    c_alpha,
                    c_sigma,
                    c_beta,
                    d_delta,
                    d_theta,
                    d_alpha,
                    d_sigma,
                    d_beta,
                    e_delta,
                    e_theta,
                    e_alpha,
                    e_sigma,
                    e_beta,
                    f_delta,
                    f_theta,
                    f_alpha,
                    f_sigma,
                    f_beta,
                    g_delta,
                    g_theta,
                    g_alpha,
                    g_sigma,
                    g_beta,
                    "label"
                    ) FROM STDIN WITH CSV
                """,
                file=sio
            )

        self.conn.commit()

    def __shuffleChunks(self):
        with self.conn.cursor() as cursor:
            cursor.execute(f"SELECT id FROM samples;")
            return shuffle(list(map(lambda x: x[0], cursor.fetchall())))
        
    def __validationChunks(self):
        with self.conn.cursor() as cursor:
            cursor.execute(f"SELECT id FROM validation;")
            return list(map(lambda x: x[0], cursor.fetchall()))
    
    def sampleNum(self, mode = "samples"):
        with self.conn.cursor() as cursor:
            cursor.execute(f"SELECT count(*) FROM {mode};")
            result = cursor.fetchone()
            return result[0]
        
    def __classWeights(self, mode = "samples"):
        weights = {}

        with self.conn.cursor() as cursor:

            cursor.execute(f"SELECT count(*) FROM {mode};")
            result = cursor.fetchone()
            total = result[0]

            for label in range(self.num_classes):
                cursor.execute(f'SELECT count(*) FROM {mode} WHERE "label" = {label};')
                result = cursor.fetchone()
                freq = result[0]

                weights[label] =  1 - (freq/total)

        return weights

    def readChunks(self, batchSize: int, epochs: int, mode = "samples"):
            
            classWeights = self.__classWeights(mode)
            
            with self.conn.cursor() as cursor:
                for _ in range(epochs):

                    available = None
                    if mode == "samples":
                        available = self.__shuffleChunks()    
                    else:
                        available = self.__validationChunks()

                    for slice in gen_batches(len(available), batchSize):
                        batch = available[slice]
                        
                        cursor.execute(f"""
                            SELECT a_delta, a_theta, a_alpha, a_sigma, a_beta, b_delta, b_theta, b_alpha, b_sigma, b_beta, c_delta, c_theta, c_alpha, c_sigma, c_beta, d_delta, d_theta, d_alpha, d_sigma, d_beta,
                                   e_delta, e_theta, e_alpha, e_sigma, e_beta, f_delta, f_theta, f_alpha, f_sigma, f_beta, g_delta, g_theta, g_alpha, g_sigma, g_beta, "label" 
                            FROM {mode} AS s
                            WHERE s.id IN %s;   
                        """,
                        (tuple(batch),))

                        df = pd.DataFrame(cursor.fetchall(), columns=[
                            "a_delta",
                            "a_theta",
                            "a_alpha",
                            "a_sigma",
                            "a_beta",
                            "b_delta",
                            "b_theta",
                            "b_alpha",
                            "b_sigma",
                            "b_beta",
                            "c_delta",
                            "c_theta",
                            "c_alpha",
                            "c_sigma",
                            "c_beta",
                            "d_delta",
                            "d_theta",
                            "d_alpha",
                            "d_sigma",
                            "d_beta",
                            "e_delta",
                            "e_theta",
                            "e_alpha",
                            "e_sigma",
                            "e_beta",
                            "f_delta",
                            "f_theta",
                            "f_alpha",
                            "f_sigma",
                            "f_beta",
                            "g_delta",
                            "g_theta",
                            "g_alpha",
                            "g_sigma",
                            "g_beta",
                            "label"
                        ])
                        
                        labels = df["label"]
                        chunks = df.drop(columns=["label"])

                        # Data is already normalized in featuresPerEvent (modules/edf.py), previously to data insertion.
                        scaler = MinMaxScaler()
                        scaled = scaler.fit_transform(chunks.transpose())
                            
                        yield tf.stack(scaled.T), tf.stack(keras.utils.to_categorical(labels, num_classes=self.num_classes)), np.vectorize(lambda x: classWeights[x])(labels)
                        
    def close(self):
        self.conn.close()
                    

