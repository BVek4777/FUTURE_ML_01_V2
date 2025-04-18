from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

def load_and_preprocess_data(file):
     data = pd.read_csv(file, encoding='latin-1')
     return data

   

# Function to preprocess data for Prophet model (Renaming columns)
def preprocess_data_for_prophet(data, ds_col, y_col):
    return data.rename(columns={ds_col: 'ds', y_col: 'y'})


# Function to perform forecasting with Prophet
def forecast_prophet(data):
    # Chronologically split the data (80% train, 20% test)
    split_index = int(len(data) * 0.8)
    train = data[:split_index]
    test = data[split_index:]

    # Initialize and fit Prophet model
    model = Prophet(growth='linear',daily_seasonality=False,weekly_seasonality=True,yearly_seasonality=True)
    model.fit(train)

    # Make future dataframe and forecast
    future_df = model.make_future_dataframe(periods=len(test))
    forecast = model.predict(future_df)

    # Extract forecasted values and actual sales for testing period
    test_forecast = forecast.iloc[-len(test):]["yhat"].values
    actual_sales = test["y"].values

    # Calculate Accuracy Metrics
    mae = mean_absolute_error(actual_sales, test_forecast)
    rmse = np.sqrt(mean_squared_error(actual_sales, test_forecast))
    r2 = r2_score(actual_sales, test_forecast)

    return forecast, mae, rmse, r2,model,test


# Function to plot forecast results
def plot_forecast_results(forecast,model,test_data):
    test_forecast = forecast.iloc[-len(test_data):]["yhat"].values
    actual_sales = test_data["y"].values

    # Plot Actual vs Predicted Sales
    plt.figure(figsize=(10, 5))
    plt.plot(test_data["ds"], actual_sales, label="Actual Sales", color="blue")
    plt.plot(test_data["ds"], test_forecast, label="Predicted Sales", linestyle="dashed", color="red")
    plt.legend()
    plt.title("Actual vs Predicted Sales")
    st.pyplot(plt)

     # Show Forecast Graph
    st.write("### 📈 Sales Forecast")
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    # Show Trend & Seasonality
    st.write("### 📊 Trend & Seasonality Analysis")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)
