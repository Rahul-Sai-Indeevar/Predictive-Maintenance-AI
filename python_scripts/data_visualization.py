import pandas as pd
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

# --- PART 1: CONNECTION ---

load_dotenv()

db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST')
db_name = os.getenv('DB_NAME')

connection_string = f'mysql+mysqlconnector://{db_user}:{db_password}@{db_host}/{db_name}'
engine = create_engine(connection_string)

# --- PART 2: DATA RETRIEVAL ---
# We pull from the "feature_engineered_data" view made in SQL
print("Fetching data from MySQL...")
df = pd.read_sql("SELECT * FROM feature_engineered_data", con=engine)

# --- PART 3: VISUALIZATION ---
def plot_engine_trend(engine_id, sensor_name):
    subset = df[df['engine_id'] == engine_id]
    plt.figure(figsize=(10, 5))
    plt.plot(subset['cycle'], subset[sensor_name], label=f'Raw {sensor_name}', alpha=0.5)
    
    # Check if the rolling average exists (the one we made in SQL)
    rolling_col = f'{sensor_name}_rolling_avg'
    if rolling_col in df.columns:
        plt.plot(subset['cycle'], subset[rolling_col], label='Rolling Trend', color='red')
        
    plt.title(f"Degradation Trend for Engine {engine_id} ({sensor_name})")
    plt.xlabel("Cycles until Failure")
    plt.ylabel("Sensor Value")
    plt.legend()
    plt.show()

# Run the plot for Engine 1 and Sensor 2
plot_engine_trend(1, 's2')


# 2. Check for "Flat" sensors
# In industrial data, some sensors don't change. They are useless for ML.
# We'll find sensors with zero variance.
stats = df.describe().T
useless_sensors = stats[stats['std'] == 0].index.tolist()
print(f"Useless constant sensors to drop: {useless_sensors}")

# 3. Visualize the Degradation (Look at Engine #1)
engine1 = df[df['engine_id'] == 1]

# compare a few sensors to see which one "reacts" to the engine dying
fig, ax = plt.subplots(1, 2, figsize=(16, 5))

ax[0].plot(engine1['cycle'], engine1['s2_rolling_avg'], color='red')
ax[0].set_title('Sensor 2 Trend (Engine 1)')
ax[0].set_xlabel('Cycles')

ax[1].plot(engine1['cycle'], engine1['s11_rolling_avg'], color='green')
ax[1].set_title('Sensor 11 Trend (Engine 1)')
ax[1].set_xlabel('Cycles')

plt.show()