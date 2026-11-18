import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / "loans.csv")

print("Dataset shape:", df.shape)
print(df.head())


df = df.dropna(subset=["anon_ssn"]).copy()

numeric_columns = [
    "loanAmount",
    "apr",
    "nPaidOff",
    "leadCost",
    "originallyScheduledPaymentAmount",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")


customer_data = df.groupby("anon_ssn").agg(
    loan_count=("loanId", "count"),
    avg_loan_amount=("loanAmount", "mean"),
    total_loan_amount=("loanAmount", "sum"),
    avg_apr=("apr", "mean"),
    avg_paid_off_count=("nPaidOff", "mean"),
    avg_lead_cost=("leadCost", "mean"),
    avg_scheduled_payment=("originallyScheduledPaymentAmount", "mean"),
)

print("\nCustomer-level data:")
print(customer_data.head())

print("\nCustomer data shape:", customer_data.shape)


features = customer_data.copy()

print("\nSelected features:")
print(features.columns.tolist())


imputer = SimpleImputer(strategy="median")

features_imputed = imputer.fit_transform(features)

print("\nMissing values handled")


scaler = StandardScaler()

features_scaled = scaler.fit_transform(features_imputed)

print("Feature scaling completed")


MAX_SAMPLE = 50000

if len(features_scaled) > MAX_SAMPLE:
    rng = np.random.RandomState(42)

    sample_indices = rng.choice(len(features_scaled), size=MAX_SAMPLE, replace=False)

    X_sample = features_scaled[sample_indices]

else:
    X_sample = features_scaled

print("\nSample size for K-Means:", X_sample.shape)


inertia_values = []
k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

    kmeans.fit(X_sample)

    inertia_values.append(kmeans.inertia_)

    print(f"K={k}, Inertia={kmeans.inertia_:.2f}")


plt.figure(figsize=(8, 5))

plt.plot(list(k_values), inertia_values, marker="o")

plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.grid(True)

plt.tight_layout()

plt.savefig(BASE_DIR / "elbow_plot.png")

plt.show()


BEST_K = 4

final_kmeans = KMeans(n_clusters=BEST_K, random_state=42, n_init=10)

final_kmeans.fit(X_sample)

customer_data["cluster"] = final_kmeans.predict(features_scaled)

print("\nClustering completed!")
print("Selected K:", BEST_K)


SILHOUETTE_SAMPLE = min(5000, len(X_sample))

silhouette_sample = X_sample[:SILHOUETTE_SAMPLE]

sample_labels = final_kmeans.predict(silhouette_sample)

score = silhouette_score(silhouette_sample, sample_labels)

print("\nSilhouette Score:", round(score, 3))


print("\nCluster Counts:")

print(customer_data["cluster"].value_counts().sort_index())


cluster_counts = customer_data["cluster"].value_counts().sort_index()

plt.figure(figsize=(8, 5))

cluster_counts.plot(kind="bar")

plt.title("Customer Distribution by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(BASE_DIR / "cluster_distribution.png")

plt.show()


cluster_profile = customer_data.groupby("cluster").agg(
    customer_count=("loan_count", "size"),
    avg_loan_count=("loan_count", "mean"),
    avg_loan_amount=("avg_loan_amount", "mean"),
    avg_total_loan=("total_loan_amount", "mean"),
    avg_apr=("avg_apr", "mean"),
    avg_paid_off=("avg_paid_off_count", "mean"),
    avg_lead_cost=("avg_lead_cost", "mean"),
    avg_scheduled_payment=("avg_scheduled_payment", "mean"),
)

print("\nCluster Profile:")
print(cluster_profile.round(2))


customer_data.to_csv(BASE_DIR / "customer_segments.csv")

cluster_profile.to_csv(BASE_DIR / "cluster_profile.csv")

print("\nFiles saved:")
print("customer_segments.csv")
print("cluster_profile.csv")
print("elbow_plot.png")
print("cluster_distribution.png")


print("\nTask 3 - Customer Segmentation Completed!")
