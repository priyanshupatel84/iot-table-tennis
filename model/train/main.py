import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
# df = pd.read_csv('TTSWING.csv')

base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, '', 'TTSWING.csv')  # adjust '..' if needed
df = pd.read_csv(file_path)


# Select only the required sensor columns
sensor_cols = ['ax_mean', 'ay_mean', 'az_mean',
               'gx_mean', 'gy_mean', 'gz_mean',
               'ax_var', 'ay_var', 'az_var',
               'gx_var', 'gy_var', 'gz_var']

X = df[sensor_cols]

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA to reduce dimensions (retain 95% variance)
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# KMeans clustering
k = 8  # 8 shot types: forehand/backhand drives, pushes, loops, blocks, lobs, smashes, drop shots
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_pca)

# Save the models
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(pca, 'pca.pkl')
joblib.dump(kmeans, 'kmeans_model.pkl')

print("✅ Model trained and saved successfully!")
