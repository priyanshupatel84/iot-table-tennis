import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import os

# Load the new test CSV file (only 6 sensor columns)
# test_df = pd.read_csv('test.csv')

base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, '', 'test.csv')  # adjust '..' if needed
test_df = pd.read_csv(file_path)


sensor_cols = ['ax_mean', 'ay_mean', 'az_mean', 'gx_mean', 'gy_mean', 'gz_mean']
X_test = test_df[sensor_cols]

# Load models
scaler = joblib.load('model/scaler.pkl')
pca = joblib.load('model/pca.pkl')
kmeans = joblib.load('model/kmeans_model.pkl')

# Extend 6D to 12D by adding 0 variance columns to match training format
for col in ['ax_var', 'ay_var', 'az_var', 'gx_var', 'gy_var', 'gz_var']:
    X_test[col] = 0.0  # assuming unknown, neutral values

X_test = X_test[['ax_mean', 'ay_mean', 'az_mean',
                 'gx_mean', 'gy_mean', 'gz_mean',
                 'ax_var', 'ay_var', 'az_var',
                 'gx_var', 'gy_var', 'gz_var']]

# Standardize and reduce dimensionality
X_scaled = scaler.transform(X_test)
X_pca = pca.transform(X_scaled)

# Predict clusters
predicted_clusters = kmeans.predict(X_pca)

# Shot label map (you can update this based on analysis of each cluster)
label_map = {
    0: 'Forehand Drive',
    1: 'Backhand Drive',
    2: 'Push',
    3: 'Loop',
    4: 'Block',
    5: 'Lob',
    6: 'Smash',
    7: 'Drop Shot'
}

# Assign predicted shot types
shot_types = [label_map.get(c, 'Unknown') for c in predicted_clusters]

# Assign ratings (based on distance to cluster center)
distances = kmeans.transform(X_pca)
min_dists = distances.min(axis=1)

# Normalize distances to rating 1–5 (lower distance = better shot)
dist_scaled = (min_dists - min_dists.min()) / (min_dists.max() - min_dists.min())
ratings = 5 - (dist_scaled * 4)  # 5 best, 1 worst

# Result summary
test_df['Predicted Shot'] = shot_types
test_df['Rating (1-5)'] = ratings.round(2)

# Show shot count summary
summary = test_df['Predicted Shot'].value_counts().reset_index()
summary.columns = ['Shot Type', 'Count']

print("🔍 Shot Count Summary:\n", summary)
print("\n⭐ Individual Ratings:\n", test_df[['Predicted Shot', 'Rating (1-5)']])

# Save results if needed
test_df.to_csv("test_results.csv", index=False)
