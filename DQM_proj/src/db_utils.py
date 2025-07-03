# src/db_utils.py

from sqlalchemy import create_engine, text
import pandas as pd

def get_db_engine(user, password, host, port, db_name):
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"
    return create_engine(url)

def get_table_names(engine):
    query = """
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
    """
    with engine.connect() as conn:
        result = conn.execute(text(query))
        return [row[0] for row in result]

def load_table_data(engine, table_name):
    query = f"SELECT * FROM {table_name} LIMIT 100"
    return pd.read_sql(query, engine)