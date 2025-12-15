# 📊 RFM Customer Segmentation with Automated Cluster Validation

## 📌 Project Overview

This project performs **end-to-end RFM (Recency, Frequency, Monetary) analysis** on a large-scale transactional retail dataset and applies **automated cluster validation** to identify meaningful customer segments.

Unlike basic RFM implementations, this project:

* Uses **robust quantile-based RFM scoring**
* Dynamically determines the **optimal number of clusters**
* Combines **Elbow Method + Silhouette Score**
* Introduces a custom **RFM Validation Index** to assess segmentation quality

The outcome is a **business-ready customer segmentation framework** suitable for CRM strategy, retention modeling, and revenue optimization.

---

## 🗂 Dataset

**Source:**
📌 *Online Retail Dataset* — Kaggle
[https://www.kaggle.com/datasets/carrie1/ecommerce-data](https://www.kaggle.com/datasets/carrie1/ecommerce-data)

**Description:**
Transactional data from a UK-based online retailer between **2009–2011**.

**Scale:**

* Raw rows: **1,067,371**
* Cleaned rows: **805,549**
* Unique customers: **5,878**

### Key Columns Used

| Column      | Description                |
| ----------- | -------------------------- |
| Invoice     | Transaction identifier     |
| InvoiceDate | Date and time of purchase  |
| Quantity    | Number of items purchased  |
| Price       | Unit price                 |
| Customer ID | Unique customer identifier |

---

## 🧹 Data Cleaning & Preparation

The following steps were applied:

* Removed transactions with missing `Customer ID`
* Filtered out negative or zero `Quantity` and `Price` (returns & errors)
* Converted `InvoiceDate` to datetime
* Created a `Revenue = Quantity × Price` feature

A **snapshot date** was defined as:

> Latest transaction date + 1 day
> Used as the reference point for Recency calculation.

---

## 📐 RFM Feature Engineering

RFM metrics were computed at the **customer level**:

| Metric        | Definition                |
| ------------- | ------------------------- |
| **Recency**   | Days since last purchase  |
| **Frequency** | Number of unique invoices |
| **Monetary**  | Total revenue generated   |

---

## 🧮 RFM Scoring & Segmentation

* Quartile-based scoring (1–4)
* Robust `safe_qcut` logic to handle skewed distributions
* Final RFM score range: **3–12**

### Segment Mapping

| RFM Score | Segment   |
| --------- | --------- |
| ≥ 10      | Champions |
| 8–9       | Loyal     |
| 6–7       | Potential |
| 4–5       | At Risk   |
| < 4       | Lost      |

### Segment Distribution

| Segment   | Customers |
| --------- | --------- |
| Champions | 1,738     |
| Loyal     | 1,190     |
| Potential | 1,220     |
| At Risk   | 1,155     |
| Lost      | 575       |

---

## 📊 Visualizations & Insights

### 1️⃣ Monetary Distribution by RFM Segment (Box Plot)

**Graph:** Box plot of `Monetary` value across RFM segments.

**Insight:**
Champions and Loyal customers show significantly higher median and upper-quartile spending, while At Risk and Lost segments have low and tightly clustered monetary values. This confirms that RFM segmentation effectively separates customers by revenue contribution.

---

### 2️⃣ Elbow Curve (Inertia vs Number of Clusters)

**Graph:** Line plot of K-Means inertia across k = 2–9.

**Insight:**
The curve shows diminishing returns after **k ≈ 4**, indicating that adding more clusters beyond this point does not substantially improve compactness. This provides an objective lower bound for cluster selection.

---

### 3️⃣ Silhouette Score Curve

**Graph:** Silhouette score plotted against number of clusters.

**Insight:**
The highest silhouette score (**0.916**) occurs at **k = 2**, indicating extremely well-separated clusters at low k. This suggests strong natural grouping in the data, particularly between high-value and low-value customers.

---

### 4️⃣ Combined Elbow & Silhouette Analysis

**Graph:** Dual-axis plot combining inertia and silhouette scores.

**Insight:**
To balance cluster separation and business interpretability, the final k was selected as a midpoint between Elbow and Silhouette recommendations, resulting in **k = 3**.

---

### 5️⃣ RFM Correlation Heatmap

**Graph:** Correlation matrix heatmap of Recency, Frequency, and Monetary.

**Insight:**

* Recency is negatively correlated with Frequency and Monetary
* Frequency and Monetary show a positive correlation

This confirms that the RFM features behave consistently with expected customer behavior patterns.

---

## 🤖 Final Clustering Results

| Cluster | Recency | Frequency | Monetary | RFM Score |
| ------- | ------- | --------- | -------- | --------- |
| 0       | 461.54  | 2.20      | 763      | 4.84      |
| 1       | 66.10   | 7.69      | 3,324    | 8.89      |
| 2       | 25.44   | 162.56    | 189,805  | 11.83     |

### Cluster Interpretation

* **Cluster 2:** Elite / Whale customers (very high frequency and revenue)
* **Cluster 1:** High-value loyal customers
* **Cluster 0:** Low-engagement, churn-risk customers

---

## 📈 Model Validation

### RFM Validation Index

A custom validation metric combining:

* **60%** correlation correctness between RFM metrics
* **40%** silhouette score quality

**Final Validation Index:**

> ✅ **96.65% — Excellent segmentation quality**

---
