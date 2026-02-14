import os
import sys
import joblib
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# 路径配置
kmeans_path = "/Dataset4D/public/mas-liu.lianlian/output/RAGesture/rl_kg/motion_upper_hands/motion_kmeans_final.pkl" # 请换成您的真实路径

def analyze_kmeans_distribution(kmeans_path):
    print(f">>> Loading K-Means from {kmeans_path}...")
    kmeans = joblib.load(kmeans_path)
    
    # 1. 检查聚类中心是否都存在
    n_clusters = kmeans.n_clusters
    print(f"    Total Clusters (K): {n_clusters}")
    
    # 2. 如果您有保存 label 的话，我们可以看分布
    # 如果没有保存 labels_ 属性 (MiniBatchKMeans 有时不会存)，我们无法直接看训练时的分布
    # 但我们可以检查 cluster_centers_ 的方差，看它们是否太接近
    
    if hasattr(kmeans, 'labels_'):
        labels = kmeans.labels_
        counts = Counter(labels)
        
        print("\n>>> Cluster Size Distribution (Top 10):")
        print(counts.most_common(10))
        
        print("\n>>> Cluster Size Distribution (Bottom 10):")
        print(counts.most_common()[:-11:-1])
        
        sizes = list(counts.values())
        print(f"\n    Max Size: {max(sizes)}")
        print(f"    Min Size: {min(sizes)}")
        print(f"    Avg Size: {np.mean(sizes):.2f}")
        
        # 检查有多少个空聚类或者极小聚类
        tiny_clusters = [k for k, v in counts.items() if v < 10]
        print(f"    Clusters with < 10 samples: {len(tiny_clusters)}")
    else:
        print("\n[!] This KMeans object doesn't store training labels. Checking centers only.")
    
    # 3. 检查聚类中心的相似度 (是否这就只是一堆重复的动作)
    centers = kmeans.cluster_centers_
    # 计算中心点之间的距离矩阵
    from sklearn.metrics.pairwise import euclidean_distances
    dists = euclidean_distances(centers)
    # 把对角线设为无穷大
    np.fill_diagonal(dists, np.inf)
    min_dists = dists.min(axis=1)
    
    print(f"\n>>> Cluster Separation Analysis:")
    print(f"    Avg distance between centers: {np.mean(dists):.4f}")
    print(f"    Min distance (closest pair): {np.min(dists):.4f}")
    
    # 如果最小距离非常小（接近0），说明聚类重叠严重，K设大了
    if np.min(dists) < 0.1:
        print("    [WARNING] Some clusters are almost identical! K might be too high.")

if __name__ == "__main__":
    analyze_kmeans_distribution(kmeans_path)