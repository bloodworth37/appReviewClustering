import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def elbow_graph(df):
    embeddings = pd.DataFrame(df['embedding'].tolist())
    sse = {}
    for i in range(1,20):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(embeddings)
        sse[i] = kmeans.inertia_

    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.title("Elbow Method Graph")
    plt.show()