from io import StringIO
import keras
import numpy as np
import pandas as pd
import psycopg2
import os
from sklearn.preprocessing import MinMaxScaler

from sklearn.utils import gen_batches, shuffle
import tensorflow as tf

class Db():

    def __init__(self):

        self.num_classes = 5

        self.db = os.getenv("DB_NAME")
        self.host = os.getenv("DB_HOST")
        self.user = os.getenv("DB_USER")
        self.pwd = os.getenv("DB_PASS")
        
        self.conn = psycopg2.connect(
            host=self.host, 
            database=self.db,
            user=self.user, 
            password=self.pwd
        )

        with self.conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS public.tags (
                    chunk_id integer NOT NULL,
                    tag smallint NOT NULL,
                    CONSTRAINT tags_pk PRIMARY KEY (chunk_id)
                );

                CREATE TABLE IF NOT EXISTS public.samples (
                    abd float8 NOT NULL,
                    c3_m2 float8 NOT NULL,
                    chest float8 NOT NULL,
                    o1_m2 float8 NOT NULL,
                    ic float8 NOT NULL,
                    e1_m2 float8 NOT NULL,
                    snore float8 NOT NULL,
                    ekg float8 NOT NULL,
                    airflow float8 NOT NULL,
                    hr float8 NOT NULL,
                    lat float8 NOT NULL,
                    rat float8 NOT NULL,
                    sao2 float8 NOT NULL,
                    c4_m1 float8 NOT NULL,
                    chin1_chin2 float8 NOT NULL,
                    e2_m1 float8 NOT NULL,
                    f4_m1 float8 NOT NULL,
                    o1_m1 float8 NOT NULL,
                    chunk_id int4 NOT NULL,
                    CONSTRAINT samples_tags_fk FOREIGN KEY (chunk_id) REFERENCES public.tags(chunk_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS public.test (
                    folder varchar NOT NULL,
                    ses varchar NOT NULL,
                    site varchar NOT NULL,
                    CONSTRAINT test_pk PRIMARY KEY (folder,ses)
                );
                           
                CREATE TABLE IF NOT EXISTS public.validation_tags (
	                chunk_id int4 NOT NULL,
	                tag int2 NOT NULL,
	                CONSTRAINT tags_validation_pk PRIMARY KEY (chunk_id)
                );
                           
                CREATE TABLE IF NOT EXISTS public.validation (
	                abd float8 NOT NULL,
	                c3_m2 float8 NOT NULL,
	                chest float8 NOT NULL,
	                o1_m2 float8 NOT NULL,
	                ic float8 NOT NULL,
	                e1_m2 float8 NOT NULL,
	                snore float8 NOT NULL,
	                ekg float8 NOT NULL,
	                airflow float8 NOT NULL,
	                hr float8 NOT NULL,
	                lat float8 NOT NULL,
	                rat float8 NOT NULL,
	                sao2 float8 NOT NULL,
	                c4_m1 float8 NOT NULL,
	                chin1_chin2 float8 NOT NULL,
	                e2_m1 float8 NOT NULL,
	                f4_m1 float8 NOT NULL,
	                o1_m1 float8 NOT NULL,
	                chunk_id int4 NOT NULL,
                    CONSTRAINT validation_tags_fk FOREIGN KEY (chunk_id) REFERENCES public.validation_tags(chunk_id) ON DELETE CASCADE
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

    def __getMaxChunk(self, mode = "samples"):
        with self.conn.cursor() as cursor:
            cursor.execute(f"SELECT MAX(chunk_id) FROM {mode};")
            result = cursor.fetchone()
            return 0 if result[0] is None else result[0]

    def flushData(self):
        with self.conn.cursor() as cursor:
            cursor.execute("TRUNCATE TABLE samples;")
            cursor.execute("TRUNCATE TABLE tags CASCADE;")
        self.conn.commit()

    def restart(self):
        with self.conn.cursor() as cursor:
            cursor.execute("TRUNCATE TABLE samples;")
            cursor.execute("TRUNCATE TABLE tags CASCADE;")
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
    def __insertChunk(self, chunk: pd.DataFrame, tag: int, chunkId: int, mode = "samples"):
        chunk = chunk.drop(columns = "time")
        chunk["chunkid"] = chunkId

        sio = StringIO()
        chunk.to_csv(sio, index=None, header=None)
        sio.seek(0)

        with self.conn.cursor() as cursor:
            cursor.execute(f"""
                INSERT INTO {"tags" if mode == "samples" else "validation_tags"} (tag, chunk_id)
                VALUES (%s, %s);
            """, 
            [tag, chunkId])

            cursor.copy_expert(
                sql = f"""
                    COPY {mode} (
                        abd,  
                        c3_m2,  
                        chest, 
                        o1_m2,
                        ic,
                        snore,
                        airflow,
                        hr,
                        sao2,
                        c4_m1,
                        chin1_chin2,
                        e1_m2,
                        e2_m1,
                        f4_m1,
                        o1_m1,
                        ekg,
                        lat,
                        rat,
                        chunk_id
                    ) FROM STDIN WITH CSV
                """,
                file=sio
            )

        self.conn.commit()
        
    def insertChunks(self, chunks: list, tags: list, mode = "samples"):
        maxChunk = self.__getMaxChunk(mode)

        offset = 1
        for chunk, tag in zip(list(map(lambda x: x.to_data_frame(), chunks)), tags):
            self.__insertChunk(chunk, tag, maxChunk + offset, mode)
            offset += 1

    def __shuffleChunks(self):
        with self.conn.cursor() as cursor:
            cursor.execute(f"SELECT DISTINCT chunk_id FROM tags;")
            return shuffle(list(map(lambda x: x[0], cursor.fetchall())))
        
    def __validationChunks(self):
        with self.conn.cursor() as cursor:
            cursor.execute(f"SELECT DISTINCT chunk_id FROM validation_tags;")
            return list(map(lambda x: x[0], cursor.fetchall()))
    
    def sampleNum(self, mode = "tags"):
        with self.conn.cursor() as cursor:
            cursor.execute(f"SELECT count(*) FROM {mode};")
            result = cursor.fetchone()
            return result[0]

    def readChunks(self, batchSize: int, epochs: int, mode = "samples"):
            
            classWeights = self.__classWeights("tags" if mode == "samples" else "validation_tags")
            
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
                            SELECT abd, c3_m2, chest, o1_m2, ic, e1_m2, snore, ekg, airflow, hr, lat, rat, sao2, c4_m1, chin1_chin2, e2_m1, f4_m1, o1_m1, t.chunk_id, tag 
                            FROM {mode} AS s LEFT JOIN {"tags" if mode == "samples" else "validation_tags"} AS t ON t.chunk_id = s.chunk_id 
                            WHERE s.chunk_id IN %s;   
                        """,
                        (tuple(batch),))

                        df = pd.DataFrame(cursor.fetchall(), columns=[
                            "abd", 
                            "c3_m2", 
                            "chest", 
                            "o1_m2", 
                            "ic", 
                            "e1_m2", 
                            "snore", 
                            "ekg", 
                            "airflow", 
                            "hr", 
                            "lat", 
                            "rat", 
                            "sao2", 
                            "c4_m1", 
                            "chin1_chin2", 
                            "e2_m1", 
                            "f4_m1", 
                            "o1_m1", 
                            "chunk_id", 
                            "tag"
                        ])
                        
                        tags = []
                        chunks = []

                        for chunkId in batch:
                            chunk = df.loc[df["chunk_id"] == chunkId]
                            tag = chunk.loc[:, "tag"].iloc[0]
                            chunk = chunk.drop(columns=["chunk_id", "tag"])
                            
                            # MinMaxScaler
                            chunk[[
                                "abd", 
                                "c3_m2", 
                                "chest", 
                                "o1_m2", 
                                "ic", 
                                "e1_m2", 
                                "snore", 
                                "ekg", 
                                "airflow", 
                                "hr", 
                                "lat", 
                                "rat", 
                                "sao2", 
                                "c4_m1", 
                                "chin1_chin2", 
                                "e2_m1", 
                                "f4_m1", 
                                "o1_m1"
                            ]] = MinMaxScaler().fit_transform(chunk[[
                                "abd", 
                                "c3_m2", 
                                "chest", 
                                "o1_m2", 
                                "ic", 
                                "e1_m2", 
                                "snore", 
                                "ekg", 
                                "airflow", 
                                "hr", 
                                "lat", 
                                "rat", 
                                "sao2", 
                                "c4_m1", 
                                "chin1_chin2", 
                                "e2_m1", 
                                "f4_m1", 
                                "o1_m1"
                            ]])

                            chunks.append(chunk)
                            tags.append(tag)

                        yield tf.stack(chunks), tf.stack(keras.utils.to_categorical(tags, num_classes=self.num_classes)), np.vectorize(lambda x: classWeights[x])(tags)

    def __classWeights(self, mode = "tags"):
        weights = {}

        with self.conn.cursor() as cursor:

            cursor.execute(f"SELECT count(*) FROM {mode};")
            result = cursor.fetchone()
            total = result[0]

            for label in range(self.num_classes):
                cursor.execute(f"SELECT count(*) FROM {mode} WHERE tag = {label};")
                result = cursor.fetchone()
                freq = result[0]

                weights[label] =  1 - (freq/total)

        return weights


    def readChannel(self, batchSize: int, epochs: int, channel: str, mode = "samples"):
            
            classWeights = self.__classWeights("tags" if mode == "samples" else "validation_tags")
            
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
                            SELECT {channel}, t.chunk_id, tag 
                            FROM {mode} AS s LEFT JOIN {"tags" if mode == "samples" else "validation_tags"} AS t ON t.chunk_id = s.chunk_id 
                            WHERE s.chunk_id IN %s;   
                        """,
                        (tuple(batch),))

                        df = pd.DataFrame(cursor.fetchall(), columns=[
                            channel, 
                            "chunk_id", 
                            "tag"
                        ])
                        
                        tags = []
                        chunks = []

                        for chunkId in batch:
                            chunk = df.loc[df["chunk_id"] == chunkId]
                            tag = chunk.loc[:, "tag"].iloc[0]
                            chunk = chunk.drop(columns=["chunk_id", "tag"])

                            # MinMaxScaler
                            chunk[[channel]] = MinMaxScaler().fit_transform(chunk[[channel]])

                            chunks.append(chunk)
                            tags.append(tag)

                        
                        yield tf.stack(chunks), tf.stack(keras.utils.to_categorical(tags, num_classes=self.num_classes)), np.vectorize(lambda x: classWeights[x])(tags)
                        
    def close(self):
        self.conn.close()
                    

