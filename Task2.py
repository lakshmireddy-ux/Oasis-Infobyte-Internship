# ============================
# Customer Segmentation - iFood Dataset
# ============================

# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 2. Load your uploaded dataset
df = pd.read_csv(r"C:\Users\laksh\OneDrive\Desktop\Internship\Oasis Infobyte\ifood_df.csv")

# 3. Basic info
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

# 4. Create RFM features
df['Frequency'] = (
    df['NumWebPurchases'] +
    df['NumCatalogPurchases'] +
    df['NumStorePurchases'] +
    df['NumDealsPurchases']
)
df['Monetary'] = df['MntTotal']
df['Recency'] = df['Recency']  # Already present

features = df[['Recency', 'Frequency', 'Monetary']]

print(features.head())

# 5. Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# 6. Find optimal k (optional, here we use k=4)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# 7. K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print("\nCluster Centers (scaled):\n", kmeans.cluster_centers_)
print("\nCluster counts:\n", df['Cluster'].value_counts())

# 8. Visualize clusters
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='Monetary', y='Frequency', hue='Cluster', palette='Set2')
plt.title('Customer Segments by Monetary vs Frequency')
plt.xlabel('Total Spend')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='Recency', y='Monetary', hue='Cluster', palette='Set1')
plt.title('Customer Segments by Recency vs Monetary')
plt.xlabel('Recency')
plt.ylabel('Total Spend')
plt.show()

# 9. Silhouette Score (optional, good to check quality)
score = silhouette_score(X_scaled, df['Cluster'])
print(f"Silhouette Score: {score:.3f}")

# 10. Recommendations (example)
print("\nSample Recommendations:")
print("- Target high Monetary + high Frequency segments for loyalty offers.")
print("- Engage low Recency + low Frequency segments with reactivation campaigns.")
print("- Use clusters to design personalized promotions.")
