""" This module transfers the clean data to an elephant sql database """
import psycopg2 
import sqlite3
import pandas as pd
from db_queries import CREATE_AIR_TABLE, GET_AIR_B

# download Final_EDA.csv from Github REPO
df = pd.read_csv("Final_EDA.csv")

# Connect to SQLite to prepare for df to sql transfer
sl_conn = sqlite3.connect("backend-model/air.sqlite3") 

# Transfer df to sql-lite
df.to_sql('air_b', con= sl_conn, if_exists='append')

# Create connections
# Connect to ELEPHANT SQL Database
pg_conn = psycopg2.connect(
        dbname="", 
        user="", 
        password="",
        host= "")

# Create Cursor to take query to database
pg_curs = pg_conn.cursor()

# Create cursor for Sqlite DB
sl_curs = sl_conn.cursor()

# Create functions to extract, transform, and load data
def get_queries(curs, query):
        """ 
        This function will query from SQlite 
        """
        results = curs.execute(query)
        fetch = results.fetchall()
        return fetch 

def get_queries_pg(pg_curs, query):
        """
        This function queries from 
        PostgreSQL
        """
        results= pg_curs.execute(query)
        pg_conn.commit()
        return results

def transfer_table(pg_curs, t_list):
        # t are rows in transfer list
        for t in t_list:
                insert_statement= """
                INSERT INTO air_b (
                index,
                property_type,
                room_type,
                accommodates,
                bathrooms,
                bed_type,
                cancellation_policy,
                cleaning_fee,
                city,
                host_identity_verified,
                host_since,
                instant_bookable,
                review_scores_rating,
                zipcode,
                bedrooms,
                beds,
                price)
                VALUES {};
                """.format(t[0:])       
                pg_curs.execute(insert_statement)
                #Commit the changes to Elephang SQL
        pg_conn.commit()


def execute_transfer():
        """
        This function will execute extract data from
        SQL Lite,transform data, and load data to 
        Elephant SQL
        """
        get_queries_pg(pg_curs=pg_curs, query=CREATE_AIR_TABLE)
        t_list = get_queries(curs=sl_curs, query=GET_AIR_B)
        transfer_table(pg_curs, t_list= t_list)

#Perform data transfer from Titanic df to Elephant SQL 
execute_transfer()