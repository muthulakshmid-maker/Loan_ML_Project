import pandas as pd
import numpy as np
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

df = pd.read_csv("loans.csv")

print("Dataset shape:", df.shape)
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())


target = "loanStatus"

df = df.dropna(subset=[target]).copy()

counts = df[target].value_counts()
rare_classes = counts[counts < 10].index

df[target] = df[target].replace(
    rare_classes, "Other/Rare"
)

y = df[target]

drop_columns = [
    "loanId",
    "anon_ssn",
    "loanStatus",
    "approved",
    "originated",
    "isFunded",
    "originatedDate",
    "fpStatus",
    "clarityFraudId"
]

X = df.drop(columns=drop_columns)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.25,
    random_state=42,
    stratify=y_train
)

print("\nTrain:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)



class LoanFeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        date = pd.to_datetime(
            X["applicationDate"],
            errors="coerce"
        )

        X["application_year"] = date.dt.year
        X["application_month"] = date.dt.month
        X["application_day"] = date.dt.dayofweek

        X["loan_payment_ratio"] = (
            X["loanAmount"] /
            X["originallyScheduledPaymentAmount"].replace(0, np.nan)
        )

        X["log_leadCost"] = np.log1p(
            X["leadCost"].clip(lower=0)
        )

        X["loanAmountBand"] = pd.cut(
            X["loanAmount"],
            bins=[-np.inf, 500, 1500, np.inf],
            labels=["Low", "Medium", "High"]
        )

        X = X.drop(columns=["applicationDate"])

        return X



numeric_features = [
    "apr",
    "nPaidOff",
    "loanAmount",
    "originallyScheduledPaymentAmount",
    "leadCost",
    "hasCF",
    "application_year",
    "application_month",
    "application_day",
    "loan_payment_ratio",
    "log_leadCost"
]

categorical_features = [
    "payFrequency",
    "state",
    "leadType"
]

ordinal_features = [
    "loanAmountBand"
]


numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

ordinal_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(
        categories=[["Low", "Medium", "High"]],
        handle_unknown="use_encoded_value",
        unknown_value=-1
    ))
])


preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
        ("ordinal", ordinal_pipeline, ordinal_features)
    ]
)



model = Pipeline(steps=[
    ("feature_engineering", LoanFeatureEngineer()),
    ("preprocessing", preprocessor),
    ("classifier", SGDClassifier(
        loss="log_loss",
        class_weight="balanced",
        max_iter=1000,
        tol=0.001,
        random_state=42
    ))
])


cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy",
    n_jobs=1
)

print("\nCross Validation Scores:", cv_scores)
print("Pipeline CV Accuracy:", cv_scores.mean())
print("CV Standard Deviation:", cv_scores.std())



model.fit(X_train, y_train)

val_accuracy = model.score(X_val, y_val)

print("\nValidation Accuracy:", val_accuracy)



joblib.dump(
    model,
    "loan_status_pipeline.pkl"
)

print("\nPipeline saved successfully!")



loaded_model = joblib.load(
    "loan_status_pipeline.pkl"
)

print("Pipeline loaded successfully!")




new_rows = pd.DataFrame([
    {
        "payFrequency": "B",
        "apr": 250,
        "nPaidOff": 1,
        "loanAmount": 500,
        "originallyScheduledPaymentAmount": 900,
        "state": "IL",
        "leadType": "lead",
        "leadCost": 5,
        "hasCF": 1,
        "applicationDate": "2016-05-10"
    },
    {
        "payFrequency": "W",
        "apr": 300,
        "nPaidOff": 0,
        "loanAmount": 1000,
        "originallyScheduledPaymentAmount": 1800,
        "state": "CA",
        "leadType": "organic",
        "leadCost": 3,
        "hasCF": 0,
        "applicationDate": "2016-06-15"
    },
    {
        "payFrequency": "M",
        "apr": 200,
        "nPaidOff": 2,
        "loanAmount": 2000,
        "originallyScheduledPaymentAmount": 3500,
        "state": "TX",
        "leadType": "prescreen",
        "leadCost": 8,
        "hasCF": 1,
        "applicationDate": "2016-07-20"
    }
])

predictions = loaded_model.predict(new_rows)

print("\nPredictions for 3 new rows:")

for i, prediction in enumerate(predictions, start=1):
    print(f"Customer {i}: {prediction}")