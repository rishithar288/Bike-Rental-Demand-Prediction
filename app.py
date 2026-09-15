import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date


# ---------------------------------------------------
# Load model and encoder
# ---------------------------------------------------

model = joblib.load("bike_rental_model.pkl")
encoder = joblib.load("bike_rental_encoder.pkl")


# ---------------------------------------------------
# Page configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Bike Rental Demand Prediction",
    page_icon="🚲",
    layout="centered"
)


# ---------------------------------------------------
# Title
# ---------------------------------------------------

st.title("🚲 Bike Rental Demand Prediction")

st.write(
    "Enter the weather, date and rental conditions "
    "to predict the bike rental demand."
)


# ---------------------------------------------------
# User Inputs
# ---------------------------------------------------

st.subheader("Enter Input Details")


# Date
selected_date = st.date_input(
    "Date",
    value=date(2012, 6, 15)
)


# Hour
hour = st.slider(
    "Hour",
    min_value=0,
    max_value=23,
    value=12
)


# Temperature
temp = st.number_input(
    "Temperature (normalized 0–1)",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01
)


# Feeling temperature
atemp = st.number_input(
    "Feeling Temperature (normalized 0–1)",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01
)


# Humidity
hum = st.number_input(
    "Humidity (normalized 0–1)",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01
)


# Windspeed
windspeed = st.number_input(
    "Windspeed (normalized 0–1)",
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.01
)


# Season
season = st.selectbox(
    "Season",
    [
        "springer",
        "summer",
        "fall",
        "winter"
    ]
)


# Holiday
holiday = st.selectbox(
    "Holiday",
    [
        "No",
        "Yes"
    ]
)


# Working day
workingday = st.selectbox(
    "Working Day",
    [
        "No work",
        "Working Day"
    ]
)


# Weather
weathersit = st.selectbox(
    "Weather Situation",
    [
        "Clear",
        "Mist",
        "Light Snow",
        "Heavy Rain"
    ]
)


# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

if st.button("🚲 Predict Bike Demand"):

    # Convert date to pandas datetime
    selected_datetime = pd.to_datetime(selected_date)

    # Year
    if selected_datetime.year == 2011:
        yr = 0
    else:
        yr = 1

    # Month
    mnth = selected_datetime.month

    # Dataset weekday:
    # Sunday = 0, Monday = 1, ..., Saturday = 6
    weekday = (selected_datetime.weekday() + 1) % 7

    # Day
    day = selected_datetime.day

    # ISO week
    weekofyear = selected_datetime.isocalendar().week

    # Weekend
    is_weekend = int(weekday in [0, 6])

    # Rush hour
    rush_hour = int(hour in [7, 8, 9, 17, 18, 19])

    # Time of day
    if hour < 6:
        time_of_day = "Night"
    elif hour < 12:
        time_of_day = "Morning"
    elif hour < 17:
        time_of_day = "Afternoon"
    elif hour < 21:
        time_of_day = "Evening"
    else:
        time_of_day = "Night"

    # Cyclical hour features
    hr_sin = np.sin(2 * np.pi * hour / 24)
    hr_cos = np.cos(2 * np.pi * hour / 24)

    # Cyclical month features
    month_sin = np.sin(2 * np.pi * mnth / 12)
    month_cos = np.cos(2 * np.pi * mnth / 12)

    # Cyclical weekday features
    weekday_sin = np.sin(2 * np.pi * weekday / 7)
    weekday_cos = np.cos(2 * np.pi * weekday / 7)

    # Temperature difference
    temp_difference = atemp - temp

    # Comfort index
    comfort_index = temp * (1 - hum)

    # Create input DataFrame
    input_data = pd.DataFrame({
        "yr": [yr],
        "mnth": [mnth],
        "hr": [hour],
        "weekday": [weekday],
        "temp": [temp],
        "atemp": [atemp],
        "hum": [hum],
        "windspeed": [windspeed],
        "day": [day],
        "weekofyear": [weekofyear],
        "is_weekend": [is_weekend],
        "rush_hour": [rush_hour],
        "hr_sin": [hr_sin],
        "hr_cos": [hr_cos],
        "month_sin": [month_sin],
        "month_cos": [month_cos],
        "weekday_sin": [weekday_sin],
        "weekday_cos": [weekday_cos],
        "temp_difference": [temp_difference],
        "comfort_index": [comfort_index],

        "season": [season],
        "holiday": [holiday],
        "workingday": [workingday],
        "weathersit": [weathersit],
        "time_of_day": [time_of_day]
    })


    # ---------------------------------------------------
    # Encode categorical variables
    # ---------------------------------------------------

    categorical_features = [
        "season",
        "holiday",
        "workingday",
        "weathersit",
        "time_of_day"
    ]

    numerical_features = [
        "yr",
        "mnth",
        "hr",
        "weekday",
        "temp",
        "atemp",
        "hum",
        "windspeed",
        "day",
        "weekofyear",
        "is_weekend",
        "rush_hour",
        "hr_sin",
        "hr_cos",
        "month_sin",
        "month_cos",
        "weekday_sin",
        "weekday_cos",
        "temp_difference",
        "comfort_index"
    ]


    # Numerical features
    X_numerical = input_data[numerical_features].reset_index(drop=True)


    # Categorical features
    X_encoded_cat = encoder.transform(
        input_data[categorical_features]
    )

    X_encoded_cat = pd.DataFrame(
        X_encoded_cat,
        columns=encoder.get_feature_names_out(
            categorical_features
        )
    )


    # Combine numerical + encoded categorical
    X_input = pd.concat(
        [X_numerical, X_encoded_cat],
        axis=1
    )


    # Make sure feature order is exactly the same
    X_input = X_input.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )


    # ---------------------------------------------------
    # Prediction
    # ---------------------------------------------------

    prediction_log = model.predict(X_input)

    # Convert log prediction back to actual bike count
    prediction = np.expm1(prediction_log[0])

    prediction = max(0, prediction)


    # ---------------------------------------------------
    # Display result
    # ---------------------------------------------------

    st.success(
        f"🚲 Predicted Bike Rental Demand: {prediction:.0f} bikes"
    )

    st.info(
        f"For {selected_date.strftime('%d-%m-%Y')} at {hour}:00, "
        f"the estimated bike demand is approximately "
        f"{prediction:.0f} bikes."
    )