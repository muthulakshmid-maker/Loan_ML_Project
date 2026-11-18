import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / "loans.csv")

print(df.head())

print("Dataset shape:", df.shape)

print("Columns:", df.columns.tolist())


default_status = [
    "External Collection",
    "Internal Collection",
    "Charged Off",
    "Settled Bankruptcy",
]

non_default_status = ["Paid Off Loan"]

df = df[df["originated"] == True].copy()

df = df[df["loanStatus"].isin(default_status + non_default_status)].copy()

df["default"] = df["loanStatus"].isin(default_status).astype(int)

print("\nDefault distribution:")
print(df["default"].value_counts())


df["applicationDate"] = pd.to_datetime(df["applicationDate"], errors="coerce")

df["application_month"] = df["applicationDate"].dt.month
df["application_day"] = df["applicationDate"].dt.day
df["application_weekday"] = df["applicationDate"].dt.weekday
df["application_hour"] = df["applicationDate"].dt.hour

y = df["default"]

drop_columns = [
    "default",
    "loanStatus",
    "loanId",
    "anon_ssn",
    "originated",
    "originatedDate",
    "approved",
    "isFunded",
    "clarityFraudId",
    "applicationDate",
    "fpStatus",
]

X = df.drop(columns=drop_columns, errors="ignore")

print("\nFeatures:")
print(X.columns.tolist())


X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
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


depths = [1, 3, 5, 10, None]

train_scores = []
validation_scores = []

for depth in depths:
    model = Pipeline(
        [
            ("preprocess", preprocessor),
            ("classifier", DecisionTreeClassifier(max_depth=depth, random_state=42)),
        ]
    )

    model.fit(X_train, y_train)

    train_prediction = model.predict(X_train)
    validation_prediction = model.predict(X_val)

    train_accuracy = accuracy_score(y_train, train_prediction)

    validation_accuracy = accuracy_score(y_val, validation_prediction)

    train_scores.append(train_accuracy)
    validation_scores.append(validation_accuracy)

    print(
        f"Depth {depth}: "
        f"Train {train_accuracy:.3f} | "
        f"Validation {validation_accuracy:.3f}"
    )


labels = ["1", "3", "5", "10", "None"]

plt.figure(figsize=(8, 5))

plt.plot(labels, train_scores, marker="o", label="Train Accuracy")

plt.plot(labels, validation_scores, marker="o", label="Validation Accuracy")

plt.title("Decision Tree Overfitting")
plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("overfitting_plot.png")
plt.show()


logistic_model = Pipeline(
    [("preprocess", preprocessor), ("classifier", LogisticRegression(max_iter=1000))]
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(logistic_model, X_train, y_train, cv=cv, scoring="accuracy")

print("\n5-Fold Cross Validation Scores:")
print(cv_scores)

print("CV Mean:", round(cv_scores.mean(), 3))

print("CV Standard Deviation:", round(cv_scores.std(), 3))


numeric_data = X.select_dtypes(include=np.number)

wrong_scaler = StandardScaler()

wrong_scaled_data = wrong_scaler.fit_transform(
    numeric_data.fillna(numeric_data.median())
)

print("\nWrong method: scaler fitted on full dataset")


correct_scaler = StandardScaler()

X_train_numeric = X_train[numeric_columns].copy()
X_val_numeric = X_val[numeric_columns].copy()
X_test_numeric = X_test[numeric_columns].copy()

train_medians = X_train_numeric.median()

X_train_numeric = X_train_numeric.fillna(train_medians)
X_val_numeric = X_val_numeric.fillna(train_medians)
X_test_numeric = X_test_numeric.fillna(train_medians)

X_train_scaled = correct_scaler.fit_transform(X_train_numeric)

X_val_scaled = correct_scaler.transform(X_val_numeric)

X_test_scaled = correct_scaler.transform(X_test_numeric)

print("Correct method: scaler fitted on train only")


problems = {
    "Spam Detection": "Supervised",
    "Customer Segmentation": "Unsupervised",
    "House Price Prediction": "Supervised",
    "Anomaly Detection": "Unsupervised",
}

print("\nML Problem Types:")

for problem, category in problems.items():
    print(problem, ":", category)


workflow = [
    "1. Define business problem",
    "2. Collect data",
    "3. Clean and preprocess data",
    "4. Perform EDA",
    "5. Split data",
    "6. Train and evaluate models",
    "7. Deploy and monitor",
]

print("\nML Workflow:")

for step in workflow:
    print(step)


print("\nTASK 1 COMPLETED")
