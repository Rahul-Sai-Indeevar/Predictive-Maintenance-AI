import pandas as pd
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST')
db_name = os.getenv('DB_NAME')

connection_string = f'mysql+mysqlconnector://{db_user}:{db_password}@{db_host}/{db_name}'
engine = create_engine(connection_string)

cols = ['engine_id','cycle','setting1','setting2','setting3'] + [f's{i}' for i in range(1,22)]
df = pd.read_csv('train_FD001.txt', sep='\s+', header=None, names=cols)

df.to_sql('raw_sensor_data', con=engine, if_exists='replace', index=False)
print("Data successfully loaded into MySQL!")