import streamlit as st
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from model.prophet_model import (
    load_and_preprocess_data,
    preprocess_data_for_prophet,
    forecast_prophet,
    plot_forecast_results,
)

from model.ml_models import train_and_evaluate_model

st.set_page_config(page_title="Sales Forecasting", layout="wide")
st.title("📈 Sales Forecasting App")

# Step 1: Upload Data
file = st.file_uploader("📂 Upload Your Sales Data (CSV)", type="csv")

if file:
    data = load_and_preprocess_data(file)
    st.session_state["data"] = data
    st.success("✅ File uploaded successfully!")
    #Content of dat
    st.subheader("Content of data")
    st.write(data.head())
    # Step 2: Select Columns
    st.subheader("🛠️ Select Columns")
    columns = data.columns.tolist()
    ds_column = st.selectbox("Select Date Column", ["None"] + columns)
    y_column = st.selectbox("Select Target Column", ["None"] + columns)

    if ds_column != "None" and y_column != "None" and ds_column != y_column:
        # Convert date column if needed
        if not is_datetime64_any_dtype(data[ds_column]):
            try:
                data[ds_column] = pd.to_datetime(data[ds_column])
                st.success(f"✅ Converted '{ds_column}' to datetime.")
            except Exception as e:
                st.error(f"❌ Error converting '{ds_column}': {e}")
                st.stop()

        # Step 3: Select Model
        st.subheader("🔍 Select Forecasting Model")
        model_choice = st.radio(
            "Choose a Model",
            ["Prophet", "Linear Regression", "Decision Tree", "Random Forest"]
        )

        if model_choice == "Prophet":
            st.header("🔮 Prophet Forecasting")
            df = data[[ds_column, y_column]].copy()
            df = preprocess_data_for_prophet(df, ds_column, y_column)

            forecast, mae, rmse, r2, model, test_data = forecast_prophet(df)

            st.markdown("### 🔢 Accuracy Metrics")
            st.metric("MAE", f"{mae:.2f}")
            st.metric("RMSE", f"{rmse:.2f}")
            st.metric("R²", f"{r2:.2f}")

            plot_forecast_results(forecast, model, test_data)

        else:
            st.header(f"🧠 {model_choice} Forecasting")
            x_dates, y_true, y_pred, mae, rmse, r2 = train_and_evaluate_model(
                model_choice, data.copy(), ds_column, y_column
            )

            result_df = pd.DataFrame({
                ds_column: x_dates,
                "Actual": y_true,
                "Predicted": y_pred
            })

            st.markdown("### 🔢 Accuracy Metrics")
            st.metric("MAE", f"{mae:.2f}")
            st.metric("RMSE", f"{rmse:.2f}")
            st.metric("R²", f"{r2:.4f}")

            st.line_chart(result_df.set_index(ds_column))

    else:
        st.info("ℹ️ Please select two different columns for Date and Target.")
else:
    st.info("📥 Please upload a CSV file to begin.")
