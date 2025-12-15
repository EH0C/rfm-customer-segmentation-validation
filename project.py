# ============================================================
# RFM ANALYSIS + AUTO CLUSTER VALIDATION
# Dataset: Online Retail (2009–2011 format)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv("online_retail_II.csv")

print(f"Raw rows: {len(df)}")

# ============================================================
# 2. CLEAN DATA
# ============================================================
df = df.dropna(subset=["Customer ID"])
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Remove returns / invalid values
df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]

# Monetary value
df["Revenue"] = df["Quantity"] * df["Price"]

print(f"Clean rows: {len(df)}\n")

# ============================================================
# 3. SNAPSHOT DATE
# ============================================================
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
print(f"Snapshot date: {snapshot_date.date()}\n")

# ============================================================
# 4. RFM AGGREGATION
# ============================================================
rfm = df.groupby("Customer ID").agg(
    Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
    Frequency=("Invoice", "nunique"),
    Monetary=("Revenue", "sum")
).reset_index()

print(f"Total customers: {len(rfm)}\n")

# ============================================================
# 5. ROBUST RFM SCORING
# ============================================================
def safe_qcut(series, q, labels):
    try:
        return pd.qcut(series, q=q, labels=labels, duplicates="drop")
    except ValueError:
        return pd.qcut(series.rank(method="first"), q=q, labels=labels)

rfm["R_Score"] = safe_qcut(rfm["Recency"], 4, [4, 3, 2, 1]).astype(int)
rfm["F_Score"] = safe_qcut(rfm["Frequency"], 4, [1, 2, 3, 4]).astype(int)
rfm["M_Score"] = safe_qcut(rfm["Monetary"], 4, [1, 2, 3, 4]).astype(int)

rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

# ============================================================
# 6. SEGMENT LABELS
# ============================================================
def label_segment(score):
    if score >= 10:
        return "Champions"
    elif score >= 8:
        return "Loyal"
    elif score >= 6:
        return "Potential"
    elif score >= 4:
        return "At Risk"
    else:
        return "Lost"

rfm["Segment"] = rfm["RFM_Score"].apply(label_segment)

print("Segment distribution:")
print(rfm["Segment"].value_counts(), "\n")

# ============================================================
# 7. SEGMENT SUMMARY
# ============================================================
print(
    rfm.groupby("Segment")[["Recency", "Frequency", "Monetary"]]
    .mean()
    .round(2),
    "\n"
)

# ============================================================
# 8. VISUALIZATION
# ============================================================
plt.figure(figsize=(8, 4))
sns.boxplot(
    x="Segment",
    y="Monetary",
    data=rfm,
    order=["Champions", "Loyal", "Potential", "At Risk", "Lost"]
)
plt.title("Monetary Distribution per RFM Segment")
plt.xticks(rotation=30)
plt.show()

# ============================================================
# 9. AUTO K-MEANS VALIDATION
# ============================================================
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(
    rfm[["Recency", "Frequency", "Monetary"]]
)

inertia = []
sil_scores = []
K_range = range(2, 10)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(rfm_scaled)
    inertia.append(km.inertia_)
    sil_scores.append(silhouette_score(rfm_scaled, labels))

# Elbow detection
knee = KneeLocator(
    list(K_range),
    inertia,
    curve="convex",
    direction="decreasing"
)
elbow_k = knee.knee

sil_k = K_range[np.argmax(sil_scores)]
best_sil = max(sil_scores)

best_k = int(np.mean([elbow_k, sil_k]))

print(f"Elbow suggested k = {elbow_k}")
print(f"Silhouette suggested k = {sil_k} (score={best_sil:.3f})")
print(f"Final chosen k = {best_k}\n")

# ============================================================
# 10. FINAL CLUSTERING
# ============================================================
kmeans_final = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)
rfm["Cluster"] = kmeans_final.fit_predict(rfm_scaled)

print(
    rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary", "RFM_Score"]]
    .mean()
    .round(2)
)

# ============================================================
# 11. RFM VALIDATION INDEX
# ============================================================
corr = rfm[["Recency", "Frequency", "Monetary"]].corr()

sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Between RFM Metrics")
plt.show()

def compute_validation_index(corr, sil):
    valid_corr = 0
    if corr.loc["Recency", "Frequency"] < 0: valid_corr += 1
    if corr.loc["Recency", "Monetary"] < 0: valid_corr += 1
    if corr.loc["Frequency", "Monetary"] > 0: valid_corr += 1

    corr_score = valid_corr / 3
    sil_score_norm = min(max(sil, 0), 1)

    return round(((corr_score * 0.6) + (sil_score_norm * 0.4)) * 100, 2)

validation_index = compute_validation_index(corr, best_sil)
print(f"\nOverall RFM Validation Index: {validation_index}%")
