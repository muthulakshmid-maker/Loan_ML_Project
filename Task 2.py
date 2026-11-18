import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / "loans.csv")

print("Dataset Shape:", df.shape)
print(df.head())


df = df.dropna(subset=["loanAmount"]).copy()

df = df[df["loanAmount"] > 0].copy()

print("\nCleaned Dataset Shape:", df.shape)


df["applicationDate"] = pd.to_datetime(df["applicationDate"], errors="coerce")

df["application_month"] = df["applicationDate"].dt.month
df["application_day"] = df["applicationDate"].dt.day
df["application_weekday"] = df["applicationDate"].dt.weekday
df["application_hour"] = df["applicationDate"].dt.hour

y = df["loanAmount"]


feature_columns = [
    "payFrequency",
    "nPaidOff",
    "state",
    "leadType",
    "leadCost",
    "hasCF",
    "application_month",
    "application_day",
    "application_weekday",
    "application_hour",
]

X = df[feature_columns].copy()

print("\nSelected Features:")
print(X.columns.tolist())


X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

print("\nData Split:")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


numeric_columns = X_train.select_dtypes(include=np.number).columns.tolist()

categorical_columns = X_train.select_dtypes(exclude=np.number).columns.tolist()


numeric_pipeline = Pipeline(
    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)


categorical_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)


preprocessor = ColumnTransformer(
    [
        ("num", numeric_pipeline, numeric_columns),
        ("cat", categorical_pipeline, categorical_columns),
    ]
)


linear_model = Pipeline([("preprocess", preprocessor), ("model", LinearRegression())])

linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_val)


tree_model = Pipeline(
    [
        ("preprocess", preprocessor),
        ("model", DecisionTreeRegressor(max_depth=5, random_state=42)),
    ]
)

tree_model.fit(X_train, y_train)

tree_predictions = tree_model.predict(X_val)


def evaluate_model(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)

    mse = mean_squared_error(actual, predicted)

    rmse = np.sqrt(mse)

    r2 = r2_score(actual, predicted)

    print(f"\n{name}")
    print("-" * 40)
    print("MAE :", round(mae, 2))
    print("MSE :", round(mse, 2))
    print("RMSE:", round(rmse, 2))
    print("R2  :", round(r2, 4))

    return {"Model": name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


linear_results = evaluate_model("Linear Regression", y_val, linear_predictions)

tree_results = evaluate_model("Decision Tree Regression", y_val, tree_predictions)

results = pd.DataFrame([linear_results, tree_results])

print("\nMODEL COMPARISON")
print(results.round(2).to_string(index=False))


comparison = pd.DataFrame(
    {
        "Actual Loan Amount": y_val.iloc[:10].values,
        "Linear Prediction": linear_predictions[:10],
        "Tree Prediction": tree_predictions[:10],
    }
)

print("\nACTUAL VS PREDICTED (First 10)")
print(comparison.round(2).to_string(index=False))


print("\nTask 2 Regression Completed!")
