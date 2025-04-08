from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Paths
base_path = os.path.join(os.path.dirname(__file__), 'model')
data_path = os.path.join(base_path, 'training_data.csv')

# Load models
scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
pca = joblib.load(os.path.join(base_path, 'pca.pkl'))
kmeans = joblib.load(os.path.join(base_path, 'kmeans_model.pkl'))

# Label mapping
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

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/train")
def train():
    return render_template("train.html")

@app.route("/save_sample", methods=["POST"])
def save_sample():
    try:
        data = request.get_json()
        shot_name = data.get("shot_name")
        features = data.get("features")  # list of dicts

        df = pd.DataFrame(features)
        df['shot_name'] = shot_name

        if os.path.exists(data_path):
            df.to_csv(data_path, mode='a', header=False, index=False)
        else:
            df.to_csv(data_path, index=False)

        return jsonify({"message": "Sample saved."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/train_model", methods=["POST"])
def train_model():
    try:
        df = pd.read_csv(data_path)
        feature_cols = ['ax_mean', 'ay_mean', 'az_mean',
                        'gx_mean', 'gy_mean', 'gz_mean',
                        'ax_var', 'ay_var', 'az_var',
                        'gx_var', 'gy_var', 'gz_var']

        X = df[feature_cols]
        scaler_new = StandardScaler()
        X_scaled = scaler_new.fit_transform(X)

        pca_new = PCA(n_components=3)
        X_pca = pca_new.fit_transform(X_scaled)

        kmeans_new = KMeans(n_clusters=8, random_state=42)
        kmeans_new.fit(X_pca)

        joblib.dump(scaler_new, os.path.join(base_path, 'scaler.pkl'))
        joblib.dump(pca_new, os.path.join(base_path, 'pca.pkl'))
        joblib.dump(kmeans_new, os.path.join(base_path, 'kmeans_model.pkl'))

        return jsonify({"message": "Model trained successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        session_data = request.get_json()

        if not isinstance(session_data, list) or len(session_data) == 0:
            return jsonify({"error": "Expected a non-empty list of session data."}), 400

        df = pd.DataFrame(session_data)

        expected_cols = ['ax_mean', 'ay_mean', 'az_mean',
                         'gx_mean', 'gy_mean', 'gz_mean',
                         'ax_var', 'ay_var', 'az_var',
                         'gx_var', 'gy_var', 'gz_var']

        for col in expected_cols:
            if col not in df.columns:
                return jsonify({"error": f"Missing column: {col}"}), 400

        df = df[expected_cols]

        X_scaled = scaler.transform(df)
        X_pca = pca.transform(X_scaled)

        predicted_clusters = kmeans.predict(X_pca)
        distances = kmeans.transform(X_pca)
        min_dists = distances.min(axis=1)

        if min_dists.max() == min_dists.min():
            ratings = np.ones(len(min_dists)) * 3
        else:
            dist_scaled = (min_dists - min_dists.min()) / (min_dists.max() - min_dists.min())
            ratings = 5 - (dist_scaled * 4)

        shot_types = [label_map.get(c, 'Unknown') for c in predicted_clusters]

        df['Predicted Shot'] = shot_types
        df['Rating (1-5)'] = ratings.round(2)

        predictions = df[['Predicted Shot', 'Rating (1-5)']].to_dict(orient='records')

        return jsonify({
            "total_shots": len(predictions),
            "predictions": predictions
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/save_training_data", methods=["POST"])
def save_training_data():
    try:
        data = request.get_json()
        if not isinstance(data, list) or not data:
            return jsonify({"error": "Invalid training data."}), 400

        df = pd.DataFrame(data)
        if 'label' not in df.columns:
            return jsonify({"error": "Missing label in training data."}), 400

        os.makedirs("training_data", exist_ok=True)
        file_path = os.path.join("training_data", "training_samples.csv")
        df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)
        return jsonify({"message": f"Saved {len(df)} samples to training_samples.csv"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/reset_session", methods=["POST"])
def reset_session():
    return jsonify({"message": "No session tracking used in this version. Data is sent directly to /predict."})

if __name__ == "__main__":
    app.run(debug=True)
