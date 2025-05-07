from io import StringIO
import keras
import numpy as np
import pandas as pd
import psycopg2
import os

from sklearn.utils import gen_batches, shuffle
import tensorflow as tf

class Db():

    def __init__(self):

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

    def __getMaxChunk(self):
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT MAX(chunk_id) FROM samples;")
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

    def __insertChunk(self, chunk: pd.DataFrame, tag: int, chunkId: int):
        chunk = chunk.drop(columns = "time")
        chunk["chunkid"] = chunkId

        sio = StringIO()
        chunk.to_csv(sio, index=None, header=None)
        sio.seek(0)

        with self.conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO tags (tag, chunk_id)
                VALUES (%s, %s);
            """, 
            [tag, chunkId])

            cursor.copy_expert(
                sql = """
                    COPY samples (
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
        
    def insertChunks(self, chunks: list, tags: list):
        maxChunk = self.__getMaxChunk()

        offset = 1
        for chunk, tag in zip(list(map(lambda x: x.to_data_frame(), chunks)), tags):
            self.__insertChunk(chunk, tag, maxChunk + offset)
            offset += 1

    def __shuffleChunks(self):
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT DISTINCT chunk_id FROM tags;")
            return shuffle(list(map(lambda x: x[0], cursor.fetchall())))
    
    def sampleNum(self):
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM tags;")
            result = cursor.fetchone()
            return result[0]

    def readChunks(self, batchSize: int, epochs: int):
            
            with self.conn.cursor() as cursor:
                for _ in range(epochs):
                    available = self.__shuffleChunks()    

                    for slice in gen_batches(len(available), batchSize):
                        batch = available[slice]
                        
                        cursor.execute("""
                            SELECT abd, c3_m2, chest, o1_m2, ic, e1_m2, snore, ekg, airflow, hr, lat, rat, sao2, c4_m1, chin1_chin2, e2_m1, f4_m1, o1_m1, t.chunk_id, tag 
                            FROM samples AS s LEFT JOIN tags AS t ON t.chunk_id = s.chunk_id 
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

                            chunks.append(chunk)
                            tags.append(tag)

                        yield tf.stack(chunks), tf.stack(keras.utils.to_categorical(tags, num_classes=5))
                
    def close(self):
        self.conn.close()
                    

