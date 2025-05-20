# AMAZON DELIVERY TIME PREDICTION PROJECT

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from geopy.distance import geodesic
import joblib
import streamlit as st
import mlflow
import mlflow.sklearn

# Step 2: Load Dataset
df = pd.read_csv("amazon_delivery.csv")

# Step 3: Data Cleaning
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')
df['Order_Time'] = pd.to_datetime(df['Order_Time'], errors='coerce').dt.hour
df['Pickup_Time'] = pd.to_datetime(df['Pickup_Time'], errors='coerce').dt.hour
df.dropna(subset=['Order_Date', 'Order_Time', 'Pickup_Time'], inplace=True)

# Step 4: Feature Engineering
def calculate_distance(row):
    store_loc = (row['Store_Latitude'], row['Store_Longitude'])
    drop_loc = (row['Drop_Latitude'], row['Drop_Longitude'])
    return geodesic(store_loc, drop_loc).km

df['Distance_km'] = df.apply(calculate_distance, axis=1)
df['Weekday'] = df['Order_Date'].dt.dayofweek

# Encode categorical features
df = pd.get_dummies(df, columns=['Weather', 'Traffic', 'Vehicle', 'Area', 'Category'], drop_first=True)

# Step 5: Feature Selection
features = ['Agent_Age', 'Agent_Rating', 'Order_Time', 'Pickup_Time', 'Distance_km', 'Weekday'] + \
           [col for col in df.columns if col.startswith(('Weather_', 'Traffic_', 'Vehicle_', 'Area_', 'Category_'))]
X = df[features]
y = df['Delivery_Time']

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Model Training and MLflow Tracking
mlflow.set_experiment("Amazon Delivery Time Prediction")

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42)
}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        mlflow.log_param("model_type", name)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        joblib.dump(model, f"{name}.pkl")
        mlflow.sklearn.log_model(model, name)

# Step 8: Streamlit App
# Save this as app.py and run with: streamlit run app.py

def predict_delivery_time(input_data):
    model = joblib.load("RandomForest.pkl")
    prediction = model.predict(pd.DataFrame([input_data]))
    return prediction[0]

# Uncomment this block if saving as a Streamlit app
# st.title("Amazon Delivery Time Predictor")
# age = st.number_input("Agent Age", 18, 65)
# rating = st.slider("Agent Rating", 1.0, 5.0)
# order_time = st.slider("Order Hour", 0, 23)
# pickup_time = st.slider("Pickup Hour", 0, 23)
# distance = st.number_input("Distance (km)", 0.0, 100.0)
# weekday = st.selectbox("Weekday", list(range(7)))

# input_dict = {
#     'Agent_Age': age,
#     'Agent_Rating': rating,
#     'Order_Time': order_time,
#     'Pickup_Time': pickup_time,
#     'Distance_km': distance,
#     'Weekday': weekday
#     # Add other encoded fields with default 0
# }
# for col in X.columns:
#     if col not in input_dict:
#         input_dict[col] = 0

# if st.button("Predict Delivery Time"):
#     output = predict_delivery_time(input_dict)
#     st.success(f"Estimated Delivery Time: {output:.2f} hours")