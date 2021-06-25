import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib as plt 
from sklearn.cluster import KMeans

df = pd.read_csv('stars.csv')

print(df.head())

fig = px.scatter(df, x = 'Size', y = 'Light')

fig.show()

X = df.iloc[:, [0, 1]].values

print(X)

wcss = []

for i in range(1, 11):
  kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)

  kmeans.fit(X)

  wcss.append(kmeans.inertia_)

plt.figure(figsize = (10, 5))

sns.lineplot(range(1, 11), wcss, marker = 'o', color = 'orange')

plt.title('Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()

kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)

y_kmeans = kmeans.fit_predict(X)

plt.figure(figsize = (15, 7))

sns.scatterplot(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], color = 'blue', label = 'cluster1')

sns.scatterplot(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], color = 'yellow', label = 'cluster2')

sns.scatterplot(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], color = 'green', label = 'cluster3')

sns.scatterplot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color = 'red', label = 'Centroids', s = 100, marker = ',')

plt.grid(False)

plt.title('Clusters of Charatceristics')

plt.xlabel('Size')

plt.ylabel('Light')

plt.legend()

plt.show()