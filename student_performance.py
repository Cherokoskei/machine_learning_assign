# Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

#  Loading the csv dataset
df = pd.read_csv('StudentsPerformance.csv',sep=",")  

# Selecting only numeric columns for clustering
data = df[['math score', 'reading score', 'writing score']]

#Here we Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
print(data_scaled)

# Creating a KMeans model with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(data_scaled)

# Adding cluster label to original dataframe
df['cluster'] = clusters
#cluster characteristics
print(df.groupby('cluster')[['math score', 'reading score', 'writing score']].mean())

# PCA to reduce dimensions for 2D visualization
pca = PCA(n_components=2)
components = pca.fit_transform(data_scaled)

df['PC1'] = components[:, 0]
df['PC2'] = components[:, 1]

# Plot clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=df, palette='Set1')
plt.title("Clusters of Students Based on Exam Performance")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.savefig('scatterplot.png')
plt.show()
