import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    LabelEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/results", exist_ok=True)


df = pd.read_csv("loans.csv")

print("Dataset shape:", df.shape)
print("\nLoan status distribution:")
print(df["loanStatus"].value_counts())



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

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
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
        ]
    )

    return Pipeline(steps=[
        ("features", LoanFeatureEngineer()),
        ("preprocessor", preprocessor)
    ])



print("\n========== CLASSIFICATION ==========")

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

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y_encoded,
    test_size=0.40,
    random_state=42,
    stratify=y_encoded
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

MAX_TRAIN_ROWS = 150000

if len(X_train) > MAX_TRAIN_ROWS:

    X_train, _, y_train, _ = train_test_split(
        X_train,
        y_train,
        train_size=MAX_TRAIN_ROWS,
        random_state=42,
        stratify=y_train
    )

print("Rows used for classifier training:", len(X_train))


logistic_pipeline = Pipeline(steps=[
    ("features", LoanFeatureEngineer()),
    ("preprocessor", create_preprocessor().named_steps["preprocessor"]),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

logistic_pipeline.fit(X_train, y_train)

logistic_predictions = logistic_pipeline.predict(X_val)

logistic_accuracy = accuracy_score(
    y_val, logistic_predictions
)

probabilities = logistic_pipeline.predict_proba(
    X_val.iloc[:5]
)

predictions = logistic_pipeline.predict(
    X_val.iloc[:5]
)

print("\nLOGISTIC REGRESSION")
print("Validation Accuracy:", logistic_accuracy)

print("\nPredict Probability for 5 rows:")
print(probabilities)

print("\nPredicted classes:")
print(label_encoder.inverse_transform(predictions))

print("\nActual classes:")
print(label_encoder.inverse_transform(y_val[:5]))


tree_pipeline = Pipeline(steps=[
    ("features", LoanFeatureEngineer()),
    ("preprocessor", create_preprocessor().named_steps["preprocessor"]),
    ("classifier", DecisionTreeClassifier(
        max_depth=10,
        class_weight="balanced",
        random_state=42
    ))
])

tree_pipeline.fit(X_train, y_train)

tree_predictions = tree_pipeline.predict(X_val)

tree_accuracy = accuracy_score(y_val, tree_predictions)

print("\nDECISION TREE")
print("Validation Accuracy:", tree_accuracy)


forest_pipeline = Pipeline(steps=[
    ("features", LoanFeatureEngineer()),
    ("preprocessor", create_preprocessor().named_steps["preprocessor"]),
    ("classifier", RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        bootstrap=True,
        oob_score=True,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42
    ))
])

forest_pipeline.fit(X_train, y_train)

forest_predictions = forest_pipeline.predict(X_val)

forest_accuracy = accuracy_score(y_val, forest_predictions)

print("\nRANDOM FOREST")
print("Validation Accuracy:", forest_accuracy)

print("OOB Score:",
      forest_pipeline.named_steps["classifier"].oob_score_)




xgb_pipeline = Pipeline(steps=[
    ("features", LoanFeatureEngineer()),
    ("preprocessor", create_preprocessor().named_steps["preprocessor"]),
    ("classifier", XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=len(label_encoder.classes_),
        eval_metric="mlogloss",
        early_stopping_rounds=20,
        n_jobs=-1,
        random_state=42
    ))
])

xgb_preprocessor = Pipeline(steps=[
    ("features", LoanFeatureEngineer()),
    ("preprocessor", create_preprocessor().named_steps["preprocessor"])
])

xgb_preprocessor.fit(X_train, y_train)

X_val_transformed = xgb_preprocessor.transform(X_val)

xgb_pipeline.fit(
    X_train,
    y_train,
    classifier__eval_set=[
        (X_val_transformed, y_val)
    ],
    classifier__verbose=False
)

xgb_predictions = xgb_pipeline.predict(X_val)

xgb_accuracy = accuracy_score(y_val, xgb_predictions)

print("\nXGBOOST")
print("Validation Accuracy:", xgb_accuracy)


results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "XGBoost"
    ],
    "Validation Accuracy": [
        logistic_accuracy,
        tree_accuracy,
        forest_accuracy,
        xgb_accuracy
    ]
})

print("\nCLASSIFICATION MODEL COMPARISON")
print(results.sort_values(
    by="Validation Accuracy",
    ascending=False
))

results.to_csv(
    "outputs/results/classification_results.csv",
    index=False
)


feature_names = (
    forest_pipeline.named_steps["preprocessor"]
    .get_feature_names_out()
)

forest_importance = (
    forest_pipeline.named_steps["classifier"]
    .feature_importances_
)

forest_features = pd.DataFrame({
    "Feature": feature_names,
    "Importance": forest_importance
}).sort_values(
    by="Importance",
    ascending=False
)

print("\nRANDOM FOREST TOP 5 FEATURES")
print(forest_features.head(5))


xgb_feature_names = (
    xgb_pipeline.named_steps["preprocessor"]
    .get_feature_names_out()
)

xgb_importance = (
    xgb_pipeline.named_steps["classifier"]
    .feature_importances_
)

xgb_features = pd.DataFrame({
    "Feature": xgb_feature_names,
    "Importance": xgb_importance
}).sort_values(
    by="Importance",
    ascending=False
)

print("\nXGBOOST TOP 5 FEATURES")
print(xgb_features.head(5))

forest_features.to_csv(
    "outputs/results/forest_feature_importance.csv",
    index=False
)

xgb_features.to_csv(
    "outputs/results/xgboost_feature_importance.csv",
    index=False
)


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

X_reg_train, X_reg_val, y_reg_train, y_reg_val = (
    train_test_split(
        X_reg,
        y_reg,
        test_size=0.20,
        random_state=42
    )
)

MAX_REG_ROWS = 150000

if len(X_reg_train) > MAX_REG_ROWS:

    X_reg_train, _, y_reg_train, _ = train_test_split(
        X_reg_train,
        y_reg_train,
        train_size=MAX_REG_ROWS,
        random_state=42
    )


def create_regression_pipeline(regressor):

    return Pipeline(steps=[
        ("features", LoanFeatureEngineer()),
        ("preprocessor", create_preprocessor().named_steps["preprocessor"]),
        ("regressor", regressor)
    ])


regression_models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1, max_iter=3000)
}

regression_results = []

fitted_regression_models = {}

for name, regressor in regression_models.items():

    pipeline = create_regression_pipeline(regressor)

    pipeline.fit(X_reg_train, y_reg_train)

    score = pipeline.score(X_reg_val, y_reg_val)

    regression_results.append({
        "Model": name,
        "R2 Score": score
    })

    fitted_regression_models[name] = pipeline

    print(name, "R2 Score:", score)


print("\nREGRESSION COEFFICIENT COMPARISON")

for name, pipeline in fitted_regression_models.items():

    coefficients = pipeline.named_steps["regressor"].coef_

    feature_names = (
        pipeline.named_steps["preprocessor"]
        .get_feature_names_out()
    )

    coefficient_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": np.asarray(coefficients).ravel()
    })

    coefficient_df["Absolute Coefficient"] = (
        coefficient_df["Coefficient"].abs()
    )

    print("\n", name)
    print(
        coefficient_df.sort_values(
            by="Absolute Coefficient",
            ascending=False
        ).head(10)
    )

lasso_coefficients = (
    fitted_regression_models["Lasso"]
    .named_steps["regressor"].coef_
)

zero_count = np.sum(np.abs(lasso_coefficients) < 1e-8)

print("\nLASSO ZERO COEFFICIENT COUNT:", zero_count)

pd.DataFrame(regression_results).to_csv(
    "outputs/results/regression_results.csv",
    index=False
)


print("\n========== K-MEANS CLUSTERING ==========")

cluster_columns = [
    "loanAmount",
    "apr",
    "nPaidOff"
]

cluster_data = df[cluster_columns].copy()

cluster_data = cluster_data.replace(
    [np.inf, -np.inf], np.nan
)

cluster_data = cluster_data.sample(
    n=min(10000, len(cluster_data)),
    random_state=42
)

cluster_imputer = SimpleImputer(strategy="median")

cluster_data_imputed = cluster_imputer.fit_transform(
    cluster_data
)

cluster_scaler = StandardScaler()

cluster_scaled = cluster_scaler.fit_transform(
    cluster_data_imputed
)


inertia_values = []
silhouette_values = []

for k in range(2, 9):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(cluster_scaled)

    inertia_values.append(kmeans.inertia_)

    silhouette = silhouette_score(
        cluster_scaled,
        labels
    )

    silhouette_values.append(silhouette)

    print(
        f"K={k}, Inertia={kmeans.inertia_:.2f}, "
        f"Silhouette={silhouette:.4f}"
    )


plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 9),
    inertia_values,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("K-Means Elbow Method")
plt.grid(True)

plt.savefig(
    "outputs/plots/kmeans_elbow.png"
)

plt.show()


plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 9),
    silhouette_values,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("K-Means Silhouette Scores")
plt.grid(True)

plt.savefig(
    "outputs/plots/kmeans_silhouette.png"
)

plt.show()


best_k = range(2, 9)[
    np.argmax(silhouette_values)
]

print("\nBest K:", best_k)

final_kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

cluster_labels = final_kmeans.fit_predict(
    cluster_scaled
)

cluster_data["Cluster"] = cluster_labels

print("\nCLUSTER SIZES")
print(cluster_data["Cluster"].value_counts().sort_index())

print("\nCUSTOMER SEGMENT PROFILES")
print(
    cluster_data.groupby("Cluster")[
        cluster_columns
    ].mean().round(2)
)

cluster_data.to_csv(
    "outputs/results/customer_segments.csv",
    index=False
)


print("\n================================")
print("TASK 3 COMPLETED SUCCESSFULLY")
print("================================")