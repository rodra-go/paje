"""Helper module for interfacing with PostgreSQL."""
import sys
import os
import time
import psycopg2 as pg2
import pandas as pd
from datetime import datetime

def connect_to_postgres():
    """Preconfigured to connect to PostgreSQL. Returns connection and cursor.
    con, cur = connect_to_postgres()
    """
    con = pg2.connect(host='this_postgres', user='postgres', database='postgres')
    return con, con.cursor()

def encode_target(_id):
    """Encode the target for a single row as a boolean value. Takes a row _id."""
    con, cur = connect_to_postgres()
    cur.execute("""SELECT _id, income_label FROM adult where _id ={}""".format(_id))
    this_id, income_label = cur.fetchone()
    assert this_id == _id
    greater_than_50k = (income_label == ' >50K')
    cur.execute("""
        BEGIN;
        UPDATE adult
        SET target = {}
        WHERE _id = {};
        COMMIT;
        """.format(greater_than_50k, _id))
    con.close()
    
def run_command(command):
    conn, cur = connect_to_postgres()
    cur.execute('BEGIN;')
    print(cur.execute(command))
    cur.execute('COMMIT;')
    conn.close()
    
def bulk_load_df(df, schema, table, temp_path = 'data/tmp/'):
    '''
    This function creates a csv file from PostgreSQL with query
    '''
    try:
        # Connect to DB
        start_time = time.time()
        conn, cur = connect_to_postgres()
        print("Connecting to Database")
        
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        
        # Write to CSV file
        temp_csv_name = '{}.{}_{}.csv'.format( schema, table, datetime.now() )
        temp_csv_path = temp_path + temp_csv_name
        print("Starting DataFrame CSV export...")
        df.to_csv(temp_csv_path, encoding='utf-8', header = True, doublequote = True, sep=',', index=False)
        print("CSV File has been created")
        
        # Truncate the target table
        cur.execute('BEGIN;')
        cur.execute("TRUNCATE {}.{};".format(schema,table))
        cur.execute('COMMIT;') 
        print("The table {}.{} has been successfully truncated.".format(schema,table))
        
        try:
            cur.execute("ALTER SEQUENCE {}.{}__id_seq RESTART;".format(schema,table))
            cur.execute('COMMIT;') 
            print("The incremental _id for table {}.{} has been reset.".format(schema,table))
        except:
            print("It wasn't possible to reset serial _id.")      
        
        # Load table from the file with header
        f = open(temp_csv_path, "r")
        cur.copy_expert("COPY {}.{}({}) FROM STDIN CSV HEADER QUOTE '\"'".format(schema,table,','.join(df.columns.values)), f)
        cur.execute("COMMIT;")
        print("The data has been succesfully loaded to table {}.{}".format(schema,table))
        
        # Closing the connection
        conn.close()
        print("DB connection closed.")
        
        # Remove temp CSV file
        print("Removing temporary files...")
        os.remove(temp_csv_path)
        
        print("Done.")
        print("Elapsed time: {} seconds".format(time.time() - start_time))

    except Exception as e:
        print("Error: {}".format(str(e)))
        sys.exit(1)
        
def load_query_to_df(sql_command):
    '''
    This function loads the results from a query to a dataframe.
    '''
    # Connect to DB
    conn, cur = connect_to_postgres()

    # Load the data
    data = pd.read_sql(sql_command, conn)
    conn.close()
    
    return data