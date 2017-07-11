"""Creating a class that will manage interactions with Postgres Database"""
from urlparse import urlparse

import pandas as pd
import psycopg2 as ps
import sqlalchemy as sa
from pandas.io.sql import SQLTable


def _execute_insert(self, conn, keys, data_iter):
    """Optional, but useful: helps Pandas write tables against Postgres much faster.
    See https://github.com/pydata/pandas/issues/8953 for more info
    """
    print("Using monkey-patched _execute_insert")
    data = [dict((k, v) for k, v in zip(keys, row)) for row in data_iter]
    conn.execute(self.insert_statement().values(data))

SQLTable._execute_insert = _execute_insert


class DBManager(object):

    def __init__(self, db_url):
        self.db_url = db_url
        result = urlparse(db_url)
        self.host = result.hostname
        self.user = result.username
        self.dbname = result.path[1:]
        self.password = result.password
        self.engine = sa.create_engine(db_url)

    def create_schema(self, schema):
        """Creates schema if does not exist"""
        conn_string = "host={0} user={1} dbname={2} password={3}".format(
            self.host, self.user, self.dbname, self.password)
        conn = ps.connect(conn_string)
        with conn:
            cur = conn.cursor()
            query = 'CREATE SCHEMA IF NOT EXISTS {schema};'.format(schema=schema)
            cur.execute(query)

    def load_table(self, table_name, schema):
        """Reads Table and stores in a Pandas Dataframe"""
        with self.engine.begin() as conn:
            df = pd.read_sql_table(table_name=table_name, con=conn, schema=schema)
            return df

    def load_query_table(self, query):
        """Reads a SQL Query and stores in a Pandas Dataframe"""
        with self.engine.begin() as conn:
            df = pd.read_sql(query, conn)
            return df

    def write_query_table(self, query):
        """Given a Create Table Query. Execute the Query to write against the DWH"""
        conn = ps.connect(self.db_url)
        with conn:
            cur = conn.cursor()
            cur.execute(query)

    def write_df_table(self, df, table_name, schema, dtype=None, if_exists='replace', index=False):
        """Writes Pandas Dataframe to Table in DB"""
        self.create_schema(schema=schema)

        with self.engine.begin() as conn:
            df.to_sql(name=table_name,
                      con=conn,
                      schema=schema,
                      dtype=dtype,
                      if_exists=if_exists,
                      index=index,
                      chunksize=1000
                     )
