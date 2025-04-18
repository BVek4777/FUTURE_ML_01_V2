import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def train_and_evaluate_model(model_name, data, ds_column, y_column):
    # Drop NA and sort by date
    data = data.dropna(subset=[ds_column, y_column])
    data = data.sort_values(by=ds_column)

    # Convert datetime to ordinal for modeling
    data["ds_ordinal"] = pd.to_datetime(data[ds_column]).map(pd.Timestamp.toordinal)
    
    X = data[["ds_ordinal"]]
    y = data[y_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Select model
    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Decision Tree":
        model = DecisionTreeRegressor()
    elif model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Train model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Return test dates for plotting
    test_dates = X_test["ds_ordinal"].map(lambda x: pd.Timestamp.fromordinal(x))

    return test_dates, y_test.values, y_pred, mae, rmse, r2
