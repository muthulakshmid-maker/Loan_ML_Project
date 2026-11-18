import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_recall_curve
)

from sklearn.dummy import DummyRegressor
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

os.makedirs("outputs/task4/plots", exist_ok=True)
os.makedirs("outputs/task4/results", exist_ok=True)


df = pd.read_csv("loans.csv")

print("Dataset shape:", df.shape)


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

        X["log_leadCost"] = np.log1p(
            X["leadCost"].clip(lower=0)
        )

        if "loanAmount" in X.columns:
            X["loan_payment_ratio"] = (
                X["loanAmount"] /
                X["originallyScheduledPaymentAmount"].replace(
                    0, np.nan
                )
            )
        else:
            X["apr_paidoff_interaction"] = (
                X["apr"] * X["nPaidOff"]
            )

        X = X.drop(columns=["applicationDate"])

        return X



def create_preprocessor():

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            make_column_selector(dtype_include=np.number)
        ),
        (
            "categorical",
            categorical_pipeline,
            make_column_selector(dtype_include=["object", "category"])
        )
    ])


def create_classifier(classifier):

    return Pipeline([
        ("features", LoanFeatureEngineer()),
        ("preprocessor", create_preprocessor()),
        ("classifier", classifier)
    ])


class_df = df.dropna(subset=["loanStatus"]).copy()

counts = class_df["loanStatus"].value_counts()

rare_classes = counts[counts < 10].index

class_df["loanStatus"] = class_df["loanStatus"].replace(
    rare_classes, "Other/Rare"
)

X = class_df.drop(columns=[
    "loanId",
    "anon_ssn",
    "loanStatus",
    "approved",
    "originated",
    "isFunded",
    "originatedDate",
    "fpStatus",
    "clarityFraudId"
])

y = class_df["loanStatus"]

encoder = LabelEncoder()
y = encoder.fit_transform(y)


X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nTrain:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

MAX_TRAIN = 30000

if len(X_train) > MAX_TRAIN:

    X_train, _, y_train, _ = train_test_split(
        X_train,
        y_train,
        train_size=MAX_TRAIN,
        random_state=42,
        stratify=y_train
    )

print("Training sample:", X_train.shape)


models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        objective="multi:softprob",
        num_class=len(encoder.classes_),
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42
    )
}

fitted_models = {}

for name, classifier in models.items():

    print("\nTraining:", name)

    pipeline = create_classifier(classifier)

    pipeline.fit(X_train, y_train)

    fitted_models[name] = pipeline


def evaluate_model(name, model, X_data, y_data):

    predictions = model.predict(X_data)

    accuracy = accuracy_score(y_data, predictions)

    precision = precision_score(
        y_data,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_data,
        predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_data,
        predictions,
        average="macro",
        zero_division=0
    )

    cm = confusion_matrix(
        y_data,
        predictions
    )

    print("\n==============================")
    print(name)
    print("==============================")

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_data,
            predictions,
            target_names=encoder.classes_,
            zero_division=0
        )
    )

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }


validation_results = []

for name, model in fitted_models.items():

    result = evaluate_model(
        name,
        model,
        X_val,
        y_val
    )

    validation_results.append(result)


validation_df = pd.DataFrame(validation_results)

print("\nVALIDATION RESULTS")
print(validation_df)

validation_df.to_csv(
    "outputs/task4/results/validation_metrics.csv",
    index=False
)


y_val_binary = label_binarize(
    y_val,
    classes=np.arange(len(encoder.classes_))
)

roc_models = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost"
]

plt.figure(figsize=(10, 7))

for name in roc_models:

    model = fitted_models[name]

    probabilities = model.predict_proba(X_val)

    fpr, tpr, thresholds = roc_curve(
        y_val_binary.ravel(),
        probabilities.ravel()
    )

    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {roc_auc:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("Multiclass ROC Curves")

plt.legend()
plt.grid()

plt.savefig(
    "outputs/task4/plots/roc_curves.png"
)

plt.show()


best_name = validation_df.loc[
    validation_df["F1 Score"].idxmax(),
    "Model"
]

best_model = fitted_models[best_name]

print("\nBest validation model:", best_name)

probabilities = best_model.predict_proba(X_val)

print("\nCLASS-WISE THRESHOLD ANALYSIS")

threshold_results = []

for class_index, class_name in enumerate(encoder.classes_):

    actual_binary = (
        y_val == class_index
    ).astype(int)

    class_probabilities = probabilities[:, class_index]

    precision, recall, thresholds = precision_recall_curve(
        actual_binary,
        class_probabilities
    )

    eligible = np.where(
        recall[:-1] >= 0.75
    )[0]

    if len(eligible) > 0:

        best_index = eligible[
            np.argmax(precision[:-1][eligible])
        ]

        threshold = thresholds[best_index]

        predicted_binary = (
            class_probabilities >= threshold
        ).astype(int)

        selected_precision = precision_score(
            actual_binary,
            predicted_binary,
            zero_division=0
        )

        selected_recall = recall_score(
            actual_binary,
            predicted_binary,
            zero_division=0
        )

        print(
            f"{class_name}: Threshold={threshold:.4f}, "
            f"Precision={selected_precision:.4f}, "
            f"Recall={selected_recall:.4f}"
        )

        threshold_results.append({
            "Class": class_name,
            "Threshold": threshold,
            "Precision": selected_precision,
            "Recall": selected_recall
        })

    else:

        print(
            f"{class_name}: No threshold achieved 75% recall"
        )

pd.DataFrame(threshold_results).to_csv(
    "outputs/task4/results/threshold_results.csv",
    index=False
)

print("\nRANDOM FOREST GRID SEARCH")

rf_pipeline = create_classifier(
    RandomForestClassifier(
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42
    )
)

rf_parameters = {

    "classifier__n_estimators": [100, 150],

    "classifier__max_depth": [10, 15],

    "classifier__min_samples_split": [2, 5]
}

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=rf_parameters,
    scoring="f1_macro",
    cv=cv,
    n_jobs=1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("\nBest Random Forest Parameters:")
print(grid_search.best_params_)

print("Best Random Forest CV Score:")
print(grid_search.best_score_)

fitted_models["Tuned Random Forest"] = grid_search.best_estimator_


print("\nXGBOOST RANDOMIZED SEARCH")

xgb_pipeline = create_classifier(
    XGBClassifier(
        objective="multi:softprob",
        num_class=len(encoder.classes_),
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42
    )
)

xgb_parameters = {

    "classifier__n_estimators": [100, 200, 300],

    "classifier__max_depth": [3, 5, 7],

    "classifier__learning_rate": [0.01, 0.05, 0.1],

    "classifier__subsample": [0.7, 0.8, 1.0],

    "classifier__colsample_bytree": [0.7, 0.8, 1.0]
}

random_search = RandomizedSearchCV(
    estimator=xgb_pipeline,
    param_distributions=xgb_parameters,
    n_iter=15,
    scoring="f1_macro",
    cv=cv,
    n_jobs=1,
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)

print("\nBest XGBoost Parameters:")
print(random_search.best_params_)

print("Best XGBoost CV Score:")
print(random_search.best_score_)

fitted_models["Tuned XGBoost"] = random_search.best_estimator_


tuning_results = []

for name in [
    "Tuned Random Forest",
    "Tuned XGBoost"
]:

    result = evaluate_model(
        name,
        fitted_models[name],
        X_val,
        y_val
    )

    tuning_results.append(result)

tuning_df = pd.DataFrame(tuning_results)

print("\nTUNED MODEL RESULTS")
print(tuning_df)

tuning_df.to_csv(
    "outputs/task4/results/tuned_model_metrics.csv",
    index=False
)


all_results = validation_df.to_dict("records")

for result in tuning_results:
    all_results.append(result)

all_results_df = pd.DataFrame(all_results)

best_row = all_results_df.loc[
    all_results_df["F1 Score"].idxmax()
]

best_model_name = best_row["Model"]

best_model = fitted_models[best_model_name]

print("\nFINAL BEST MODEL:", best_model_name)
print("Validation F1:", best_row["F1 Score"])


print("\n========== REGRESSION ==========")

reg_df = df.dropna(subset=["loanAmount"]).copy()

X_reg = reg_df.drop(columns=[
    "loanId",
    "anon_ssn",
    "loanStatus",
    "loanAmount",
    "approved",
    "originated",
    "isFunded",
    "originatedDate",
    "fpStatus",
    "clarityFraudId"
])

y_reg = reg_df["loanAmount"]

X_reg_train, X_reg_test, y_reg_train, y_reg_test = (
    train_test_split(
        X_reg,
        y_reg,
        test_size=0.20,
        random_state=42
    )
)


reg_preprocessor = ColumnTransformer([
    (
        "numeric",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]),
        make_column_selector(dtype_include=np.number)
    ),
    (
        "categorical",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        make_column_selector(dtype_include=["object", "category"])
    )
])


regression_pipeline = Pipeline([
    ("features", LoanFeatureEngineer()),
    ("preprocessor", reg_preprocessor),
    ("regressor", Ridge(alpha=1.0))
])

regression_pipeline.fit(
    X_reg_train,
    y_reg_train
)

reg_predictions = regression_pipeline.predict(
    X_reg_test
)


baseline = DummyRegressor(strategy="mean")

baseline.fit(
    X_reg_train,
    y_reg_train
)

baseline_predictions = baseline.predict(
    X_reg_test
)


def regression_metrics(actual, predicted):

    mae = mean_absolute_error(
        actual, predicted
    )

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    r2 = r2_score(
        actual, predicted
    )

    return mae, rmse, r2


model_metrics = regression_metrics(
    y_reg_test,
    reg_predictions
)

baseline_metrics = regression_metrics(
    y_reg_test,
    baseline_predictions
)

print("\nRIDGE REGRESSION")
print("MAE:", model_metrics[0])
print("RMSE:", model_metrics[1])
print("R2:", model_metrics[2])

print("\nMEAN BASELINE")
print("MAE:", baseline_metrics[0])
print("RMSE:", baseline_metrics[1])
print("R2:", baseline_metrics[2])

regression_comparison = pd.DataFrame({
    "Model": ["Ridge Regression", "Mean Baseline"],
    "MAE": [model_metrics[0], baseline_metrics[0]],
    "RMSE": [model_metrics[1], baseline_metrics[1]],
    "R2": [model_metrics[2], baseline_metrics[2]]
})

print("\nREGRESSION COMPARISON")
print(regression_comparison)

regression_comparison.to_csv(
    "outputs/task4/results/regression_comparison.csv",
    index=False
)

print("\n========== FINAL TEST EVALUATION ==========")

test_predictions = best_model.predict(X_test)

print("Best Model:", best_model_name)

print(
    "Test Accuracy:",
    accuracy_score(y_test, test_predictions)
)

print(
    "Test Precision:",
    precision_score(
        y_test,
        test_predictions,
        average="macro",
        zero_division=0
    )
)

print(
    "Test Recall:",
    recall_score(
        y_test,
        test_predictions,
        average="macro",
        zero_division=0
    )
)

print(
    "Test F1:",
    f1_score(
        y_test,
        test_predictions,
        average="macro",
        zero_division=0
    )
)

print("\nTest Confusion Matrix:")
print(confusion_matrix(y_test, test_predictions))


joblib.dump(
    best_model,
    "outputs/task4/best_loan_status_model.pkl"
)

print("\nBest model saved successfully!")

print("\n====================================")
print("TASK 4 COMPLETED SUCCESSFULLY")
print("====================================")
