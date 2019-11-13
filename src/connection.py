from sqlalchemy import create_engine
import pandas as pd

def create_connection(db_name):
    engine = create_engine('mysql://root:password@localhost/'+db_name)
    conn = engine.connect()
    return conn