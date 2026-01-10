import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib

# 1. Load data from MySQL

load_dotenv()

db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST')
db_name = os.getenv('DB_NAME')

connection_string = f'mysql+mysqlconnector://{db_user}:{db_password}@{db_host}/{db_name}'
engine = create_engine(connection_string)
df = pd.read_sql("SELECT * FROM feature_engineered_data", con=engine)

# 2. Define Features and Target
# We include the raw sensors AND the rolling averages we made
features = [
    's2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21',
    's2_rolling_avg', 's4_rolling_avg', 's11_rolling_avg'
]
target = 'RUL'

X = df[features]
y = df[target]

# 3. Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 4. Initialize and Train the Model
print("Training Random Forest Model... please wait.")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate the Model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f"Mean Absolute Error: {mae:.2f} cycles")
print(f"R2 Score: {r2:.2f}")

# Before training, clip the target variable
# This often boosts R2 score because it stops the model from trying to find "failure patterns" in perfectly healthy engines.
y_train_clipped = np.minimum(y_train, 125)

# Re-train with the clipped labels
model.fit(X_train, y_train_clipped)

# Check new R2 score
y_pred_clipped = model.predict(X_test)
new_r2 = r2_score(np.minimum(y_test, 125), y_pred_clipped)
print(f"New Clipped R2 Score: {new_r2:.2f}")

# 6. Plot Feature Importance
importances = model.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(10,6))
plt.title('Which Sensors Predict Failure Best?')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

# --- COST PARAMETERS ---
COST_FAILURE = 1_000_000  
COST_MAINTENANCE = 100_000 

# 1. Create the analysis dataframe
analysis_df = X_test.copy()

# 2. Pull the engine_id and actual RUL back using the index
# We look at the original 'df' and grab the IDs for the rows that ended up in X_test
analysis_df['engine_id'] = df.loc[X_test.index, 'engine_id']
analysis_df['actual_RUL'] = y_test
analysis_df['pred_RUL'] = y_pred_clipped

# --- STRATEGY 1: REACTIVE ---
# How many unique engines are we looking at in this test set?
num_engines_in_test = analysis_df['engine_id'].nunique() 

# In a reactive world, every one of these engines eventually breaks.
reactive_cost = num_engines_in_test * COST_FAILURE

# --- STRATEGY 2: PREDICTIVE ---
# We trigger maintenance when Predicted RUL <= 30
analysis_df['maintenance_triggered'] = analysis_df['pred_RUL'] <= 30

# Calculate the costs
# Total Engines we successfully caught before they hit RUL 0
engines_saved = analysis_df[analysis_df['maintenance_triggered'] == True]['engine_id'].nunique()

# Total Engines that hit RUL 0 without us triggering an alarm (The "Oops" factor)
# This is where actual_RUL is 0 but maintenance_triggered was never True for that engine
missed_engines = num_engines_in_test - engines_saved

predictive_cost = (engines_saved * COST_MAINTENANCE) + (missed_engines * COST_FAILURE)

print(f"Number of Engines in Test Set: {num_engines_in_test}")
print(f"Engines Saved by AI: {engines_saved}")
print(f"Engines Missed (Failed): {missed_engines}")
print("-" * 30)
print(f"REACTIVE STRATEGY COST: ${reactive_cost:,.2f}")
print(f"PREDICTIVE STRATEGY COST: ${predictive_cost:,.2f}")
print(f"TOTAL SAVINGS: ${reactive_cost - predictive_cost:,.2f}")

strategies = ['Reactive (Run-to-Failure)', 'Predictive (Our AI)']
costs = [reactive_cost, predictive_cost]

plt.figure(figsize=(10, 6))
bars = plt.bar(strategies, costs, color=['red', 'green'])
plt.ylabel('Total Cost ($)')
plt.title('Potential Annual Maintenance Cost Savings')

# Add labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 50000, f'${yval:,.0f}', ha='center', fontweight='bold')

plt.show()

import joblib

# After model is trained (model.fit)
# Save the model to a file
joblib.dump(model, 'pdm_model.pkl')
print("Model saved as pdm_model.pkl")