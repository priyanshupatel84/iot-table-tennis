from flask import Flask, request, jsonify, render_template, send_file
import joblib
import pandas as pd
import os
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import io
import time
import shutil
import sys
import logging
from werkzeug.utils import secure_filename
import traceback

app = Flask(__name__)

# Paths
base_path = os.path.join(os.path.dirname(__file__), 'model')
personal_data_path = os.path.join(base_path, 'personal_shot.csv')
data_path = os.path.join(base_path, 'training_data.csv')

# Load models
scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
pca = joblib.load(os.path.join(base_path, 'pca.pkl'))
kmeans = joblib.load(os.path.join(base_path, 'kmeans_model.pkl'))

# Enable uploads directory
app.config['UPLOAD_FOLDER'] = 'uploads/'

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

@app.route("/personal")
def personal():
    return render_template("personal.html")

@app.route("/predict_personal", methods=["POST"])
def predict_personal():
    """Predict shot types from sensor data for personal model"""
    try:
        data = request.get_json()
        
        if not data or not isinstance(data, list) or len(data) == 0:
            return jsonify({"error": "Invalid or empty data provided."}), 400
        
        # Load the personal model and scaler
        model_path = os.path.join(base_path, 'personal_model.h5')
        scaler_path = os.path.join(base_path, 'personal_scaler.pkl')
        encoder_path = os.path.join(base_path, 'personal_encoder.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(encoder_path):
            return jsonify({"error": "Model or necessary files not found. Please train the model first."}), 404
        
        # Load model, scaler and encoder
        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
        label_encoder = joblib.load(encoder_path)
        
        # Process each data point
        all_features = []
        for point in data:
            if 'ax' in point and 'ay' in point and 'az' in point and 'gx' in point and 'gy' in point and 'gz' in point:
                all_features.append([
                    point['ax'], point['ay'], point['az'],
                    point['gx'], point['gy'], point['gz']
                ])
        
        if not all_features:
            return jsonify({"error": "No valid data points found"}), 400
        
        # Convert to numpy array and scale
        X = np.array(all_features)
        X_scaled = scaler.transform(X)
        
        # Make predictions
        predictions = []
        for i in range(len(X_scaled)):
            # Reshape for single instance prediction
            X_single = X_scaled[i].reshape(1, -1)
            prediction_probs = model.predict(X_single, verbose=0)[0]
            prediction_class = np.argmax(prediction_probs)
            confidence = float(prediction_probs[prediction_class])
            predicted_shot = label_encoder.inverse_transform([prediction_class])[0]
            
            # Rate the shot based on confidence
            rating = min(5, max(1, int(confidence * 5)))
            
            predictions.append({
                "Predicted Shot": predicted_shot,
                "Rating (1-5)": rating,
                "Confidence": confidence
            })
        
        return jsonify({
            "message": "Predictions complete",
            "total_shots": len(predictions),
            "predictions": predictions
        }), 200
        
    except Exception as e:
        print(f"Personal prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# TensorFlow model conversion utilities
def keras_to_tflite(bin_file, quantize=False, quantize_data=None):
    """Convert Keras model to TFLite format with optional quantization"""
    # Convert the model
    model = tf.keras.models.load_model(bin_file)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        print('QUANTIZE: TRUE')
        
        def representative_dataset_generator():
            for value in quantize_data:
                # Each scalar value must be inside of a 2D array that is wrapped in a list
                yield [np.array(value, dtype=np.float32, ndmin=2)]
                
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_generator
    else:
        print('QUANTIZE: FALSE')
        
    # Enable resource variables for LSTM layers
    converter.experimental_enable_resource_variables = True
    
    # Add Select TF Ops to handle LSTM operations
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    
    # Disable lower tensor list ops as mentioned in the error
    converter._experimental_lower_tensor_list_ops = False
    
    tflite_model = converter.convert()

    base_path = os.path.splitext(bin_file)[0]
    out_file = base_path + ".tflite"

    # Save the TF Lite model
    with tf.io.gfile.GFile(out_file, 'wb') as f:
        f.write(tflite_model)
    return out_file

def tflite_to_c(tflite_file, out_folder):
    """Convert TFLite model to C array for embedding in microcontrollers"""
    out_path = os.path.join(out_folder, 'model.h')

    c_array = "const unsigned char model[] = {\n"

    with open(tflite_file, 'rb') as f:
        model_data = f.read()
        
    hex_data = ''
    for i, byte in enumerate(model_data):
        if i % 12 == 0:
            hex_data += '\n  '
        hex_data += f'0x{byte:02x}, '
        
    c_array += hex_data
    c_array += "\n};\n\n"
    c_array += f"const unsigned int model_len = {len(model_data)};"

    with open(out_path, 'w') as f:
        f.write(c_array)

    return c_array

# Function to validate shot data
def is_valid_shot(data):
    """Determine if the sensor data represents a valid shot based on thresholds"""
    MIN_SHOT_INTENSITY = 0.2  # Minimum acceleration to be considered a shot
    
    if not data or len(data) < 3:  # Need at least 3 data points
        return False
    
    # Calculate max acceleration magnitude
    max_magnitude = 0
    for point in data:
        ax, ay, az = point.get('ax', 0), point.get('ay', 0), point.get('az', 0)
        magnitude = np.sqrt(ax**2 + ay**2 + az**2)
        max_magnitude = max(max_magnitude, magnitude)
    
    # Check intensity - Maximum acceleration must exceed threshold
    if max_magnitude < MIN_SHOT_INTENSITY:
        print(f"Shot rejected - max magnitude {max_magnitude} below threshold {MIN_SHOT_INTENSITY}")
        return False
    
    # Accept all shots that exceed the threshold
    return True

@app.route("/save_personal_shot", methods=["POST"])
def save_personal_shot():
    """Save personal shot data to CSV - each data point is saved as a separate row"""
    data = request.get_json()
    shot_name = data.get('shot_name')
    features = data.get('features')
    
    # Basic validation of input data
    if not shot_name or not features:
        return jsonify({"message": "Invalid data. Missing shot name or features.", "valid": False}), 400
    
    # Validate the shot data pattern
    if not is_valid_shot(features):
        print(f"Shot rejected: {shot_name}")
        return jsonify({"message": "Data rejected - not a valid shot pattern", "valid": False}), 200
    
    # Create DataFrame from all data points
    rows = []
    for point in features:
        rows.append({
            'shot_name': shot_name,
            'ax': point.get('ax', 0),
            'ay': point.get('ay', 0),
            'az': point.get('az', 0),
            'gx': point.get('gx', 0),
            'gy': point.get('gy', 0),
            'gz': point.get('gz', 0)
        })
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Check if file exists and standardize existing columns if needed
    file_exists = os.path.isfile(personal_data_path)
    if file_exists:
        # Read existing data
        existing_data = pd.read_csv(personal_data_path)
        
        # Standardize columns: Convert 'ratings' to 'rating' if it exists
        if 'ratings' in existing_data.columns and 'rating' not in existing_data.columns:
            existing_data.rename(columns={'ratings': 'rating'}, inplace=True)
            # Save the standardized data back
            existing_data.to_csv(personal_data_path, index=False)
            print("Standardized CSV: 'ratings' column renamed to 'rating'")
    
    # Append to CSV
    df.to_csv(personal_data_path, mode='a', header=not file_exists, index=False)
    print(f"Saved {len(features)} data points with label '{shot_name}'")
    
    return jsonify({"message": f"Saved {len(features)} data points with label '{shot_name}'.", "valid": True}), 200

@app.route("/train_personal_model", methods=["POST"])
def train_personal_model():
    """Train model on personal data and convert to TFLite format"""
    try:
        # Check if data file exists
        if not os.path.exists(personal_data_path):
            return jsonify({"error": "No training data found. Record some shots first."}), 400
            
        # Load the data from CSV
        try:
            data = pd.read_csv(personal_data_path)
            print(f"Successfully loaded training data with {len(data)} samples")
        except Exception as e:
            print(f"Error reading CSV: {str(e)}")
            return jsonify({"error": f"Error reading training data: {str(e)}"}), 500
        
        # Standardize rating column names
        if 'ratings' in data.columns and 'rating' not in data.columns:
            data.rename(columns={'ratings': 'rating'}, inplace=True)
            # Save standardized data
            data.to_csv(personal_data_path, index=False)
            print("Standardized CSV: 'ratings' column renamed to 'rating'")
        
        # Check if there's enough data
        if len(data) < 10:  # Need at least 10 data points for training
            return jsonify({"error": f"Not enough training data. Found {len(data)} samples, need at least 10."}), 400
        
        # Check if there are at least 1 different shot type (class)
        shot_types = data['shot_name'].unique()
        if len(shot_types) < 1:
            return jsonify({"error": f"Need at least 1 shot type. Currently have: {', '.join(shot_types)}"}), 400
            
        print(f"Found {len(shot_types)} unique shot types: {', '.join(shot_types)}")
        
        # Get feature columns (all columns except shot_name and rating)
        feature_cols = [col for col in data.columns if col != 'shot_name' and col != 'rating']
        
        if not feature_cols:
            return jsonify({"error": "No valid feature columns found in data."}), 400
            
        print(f"Using feature columns: {', '.join(feature_cols)}")
        
        # Feature columns
        X = data[feature_cols]
        
        # Target column
        y = data['shot_name'].str.strip()  # Strip whitespace from shot names
        
        try:
            # Scale features for better model performance
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Save the scaler for later use in prediction
            joblib.dump(scaler, os.path.join(base_path, 'personal_scaler.pkl'))
            print("Scaler saved successfully")
        except Exception as e:
            print(f"Error in scaling features: {str(e)}")
            return jsonify({"error": f"Error in scaling features: {str(e)}"}), 500
        
        try:
            # Create a label encoder to convert shot types to numbers
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            
            # Save the encoder for prediction
            joblib.dump(label_encoder, os.path.join(base_path, 'personal_encoder.pkl'))
            print("Label encoder saved successfully")
        except Exception as e:
            print(f"Error in encoding labels: {str(e)}")
            return jsonify({"error": f"Error in encoding labels: {str(e)}"}), 500
        
        try:
            # Create a simple sequential model with fewer parameters to avoid memory issues
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(32, activation='relu', input_shape=(X.shape[1],)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(len(shot_types), activation='softmax')
            ])
            
            # Compile the model with a simpler optimizer
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print("Model created and compiled successfully")
            
            # Set a callback to prevent the training from taking too long
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            
            # Train the model with fewer epochs and a smaller batch size
            history = model.fit(
                X_scaled, y_encoded,
                epochs=15,  # Reduced epochs
                batch_size=4,  # Smaller batch size
                validation_split=0.2,
                verbose=1,
                callbacks=[early_stopping]
            )
            
            print("Model training completed successfully")
            
            # Get training metrics
            loss_value = history.history['loss'][-1]
            accuracy = history.history['accuracy'][-1]
            
            # Handle NaN values for JSON response
            if np.isnan(loss_value):
                loss_value = 0.0
            if np.isnan(accuracy):
                accuracy = 0.0
                
            print(f"Training metrics: loss={loss_value}, accuracy={accuracy}")
            
            # Save the Keras model
            model_path = os.path.join(base_path, 'personal_model.h5')
            model.save(model_path)
            print(f"Model saved to {model_path}")
            
            # Don't try to create TFLite model in the same request to avoid timeouts
            # Just report success and let the user convert to TFLite separately if needed
            return jsonify({
                "message": "Model trained successfully!", 
                "accuracy": float(accuracy * 100),  # Convert to percentage
                "loss": float(loss_value),
                "tflite_status": "Training successful, but TFLite conversion skipped for performance. Use /to-tflite endpoint to convert."
            }), 200
        except Exception as e:
            print(f"Error in model training: {str(e)}")
            error_msg = str(e)
            # Return a more detailed error message
            return jsonify({
                "error": f"Error in model training: {error_msg}",
                "details": "The server encountered an error during model training. Try again with fewer data points or simplify your model."
            }), 500
            
    except Exception as e:
        error_msg = str(e)
        print(f"Training error: {error_msg}")
        traceback.print_exc()  # Print full stack trace to server logs
        return jsonify({"error": error_msg}), 500

@app.route("/predict_shot", methods=["POST"])
def predict_shot():
    """Predict the shot type from sensor data and save test data to session.json"""
    try:
        data = request.get_json()
        use_session_json = data.get('use_session_json', False)
        
        # Load sensor data either from session.json or directly from request
        if use_session_json:
            try:
                with open('session.json', 'r') as f:
                    session_data = json.load(f)
                features = session_data.get('features', [])
            except (FileNotFoundError, json.JSONDecodeError):
                return jsonify({"error": "No test data found. Record a shot first."}), 400
        else:
            features = data.get('features', [])
            
        if not features or len(features) < 3:
            return jsonify({"error": "Not enough data for prediction"}), 400
        
        # Check if model exists
        model_path = os.path.join(base_path, 'personal_model.h5')
        encoder_path = os.path.join(base_path, 'personal_encoder.pkl')
        
        if not os.path.exists(model_path):
            return jsonify({"error": "No trained model found. Train a model first."}), 400
            
        if not os.path.exists(encoder_path):
            return jsonify({"error": "Label encoder not found. Train a model first."}), 400
        
        # Process the shot data - use all data points for prediction
        # Create a matrix of features from all data points
        data_points = []
        for point in features:
            if all(k in point for k in ['ax', 'ay', 'az', 'gx', 'gy', 'gz']):
                data_points.append([
                    point.get('ax', 0), 
                    point.get('ay', 0), 
                    point.get('az', 0), 
                    point.get('gx', 0), 
                    point.get('gy', 0), 
                    point.get('gz', 0)
                ])
        
        if not data_points:
            return jsonify({"error": "No valid data points in session"}), 400
            
        X = np.array(data_points)
        
        # Calculate average of all data points for a single prediction
        X_mean = np.mean(X, axis=0).reshape(1, -1)
        
        # Load the scaler and transform input
        scaler = joblib.load(os.path.join(base_path, 'personal_scaler.pkl'))
        X_scaled = scaler.transform(X_mean)
        
        # Load the model and label encoder
        model = tf.keras.models.load_model(model_path)
        label_encoder = joblib.load(encoder_path)
        
        # Make prediction (get class probabilities)
        prediction_probs = model.predict(X_scaled)[0]
        prediction_class = np.argmax(prediction_probs)
        confidence = float(prediction_probs[prediction_class])
        predicted_shot = label_encoder.inverse_transform([prediction_class])[0]
        
        print(f"Shot predicted: {predicted_shot} with confidence {confidence:.2f}")
        
        # Create probabilities dictionary for all shot types
        shot_probabilities = {}
        for i, prob in enumerate(prediction_probs):
            shot_type = label_encoder.inverse_transform([i])[0]
            shot_probabilities[shot_type] = float(prob)
            
        return jsonify({
            "prediction": predicted_shot,
            "confidence": confidence,
            "probabilities": shot_probabilities,
            "session_saved": True
        }), 200
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/to-tflite", methods=["POST", "GET"])
def to_tflite():
    """Convert an existing model to TFLite format"""
    if request.method == "GET":
        # Handle direct browser access
        model_path = os.path.join(base_path, 'personal_model.h5')
        tflite_path = model_path.replace('.h5', '.tflite')
        
        if not os.path.exists(tflite_path):
            if not os.path.exists(model_path):
                return "No trained model found. Please train a model first.", 404
                
            # TFLite model doesn't exist but H5 does - try to convert
            try:
                converter = tf.lite.TFLiteConverter.from_keras_model(
                    tf.keras.models.load_model(model_path)
                )
                tflite_model = converter.convert()
                
                with tf.io.gfile.GFile(tflite_path, 'wb') as f:
                    f.write(tflite_model)
            except Exception as e:
                return f"Error converting model to TFLite: {str(e)}", 500
        
        # Return the file for download
        try:
            return send_file(
                tflite_path, 
                mimetype='application/octet-stream',
                as_attachment=True,
                download_name='tennis_shot_model.tflite'
            )
        except Exception as e:
            return f"Error sending file: {str(e)}", 500

    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    try:
        model_path = os.path.join(base_path, 'personal_model.h5')
        if not os.path.exists(model_path):
            return jsonify({"error": "No trained model found."}), 400
            
        # Generate quantization data
        data = pd.read_csv(personal_data_path)
        
        # Standardize rating column names
        if 'ratings' in data.columns and 'rating' not in data.columns:
            data.rename(columns={'ratings': 'rating'}, inplace=True)
            # Save standardized data
            data.to_csv(personal_data_path, index=False)
            print("Standardized CSV: 'ratings' column renamed to 'rating'")
        
        # Get feature columns (all columns except shot_name and rating)
        feature_cols = [col for col in data.columns if col != 'shot_name' and col != 'rating']
        
        # Feature data
        X = data[feature_cols]
            
        scaler = joblib.load(os.path.join(base_path, 'personal_scaler.pkl'))
        X_scaled = scaler.transform(X)
        
        quantize_data = []
        for i in range(min(10, len(X_scaled))):
            quantize_data.append(X_scaled[i])
            
        tflite_path = keras_to_tflite(model_path, quantize=True, quantize_data=quantize_data)
        
        # Return the file for download
        return_data = io.BytesIO()
        with open(tflite_path, 'rb') as fo:
            return_data.write(fo.read())
        return_data.seek(0)
        
        timestr = time.strftime("%Y%m%d-%H%M%S")
        output_filename = 'model-' + timestr + '.tflite'
        
        return send_file(return_data, mimetype='application/octet-stream', 
                         download_name=output_filename, as_attachment=True)
    except Exception as e:
        print(f"TFLite conversion error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/standardize_csv", methods=["GET"])
def standardize_csv():
    """Standardize the CSV file to use a consistent 'rating' column"""
    try:
        if not os.path.exists(personal_data_path):
            return jsonify({"message": "No CSV file found."}), 404
            
        # Load data
        data = pd.read_csv(personal_data_path)
        
        # Check if both columns exist
        has_ratings = 'ratings' in data.columns
        has_rating = 'rating' in data.columns
        
        # Standardize the columns
        if has_ratings and has_rating:
            # Both columns exist, merge them
            # If rating is not NaN, use it, otherwise use ratings
            data['rating'] = data['rating'].fillna(data['ratings'])
            data = data.drop(columns=['ratings'])
            message = "Merged 'ratings' and 'rating' columns"
        elif has_ratings and not has_rating:
            # Only 'ratings' exists, rename it
            data = data.rename(columns={'ratings': 'rating'})
            message = "Renamed 'ratings' column to 'rating'"
        elif not has_ratings and has_rating:
            # Only 'rating' exists, already standardized
            message = "CSV already uses standardized 'rating' column"
        else:
            # Neither column exists, nothing to do
            message = "No rating columns found in the CSV"
        
        # Save the standardized data
        data.to_csv(personal_data_path, index=False)
        
        return jsonify({
            "message": f"CSV standardized successfully. {message}",
            "rows": len(data)
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/start_session", methods=["POST"])
def start_session():
    """Start a recording session and save sensor data to personal_shot.csv"""
    try:
        data = request.get_json()
        shot_name = data.get('shot_name')
        threshold = data.get('threshold', 0.2)  # Default threshold 0.2g
        
        if not shot_name:
            return jsonify({"error": "Shot name is required"}), 400
            
        # Clean the shot name (remove trailing spaces)
        shot_name = shot_name.strip()
        
        # Return success to indicate the session has started
        return jsonify({
            "message": f"Session started for shot type: {shot_name}",
            "shot_name": shot_name,
            "threshold": threshold,
            "status": "recording"
        }), 200
        
    except Exception as e:
        print(f"Session start error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/record_session_data", methods=["POST"])
def record_session_data():
    """Record sensor data during an active session"""
    try:
        data = request.get_json()
        shot_name = data.get('shot_name')
        sensor_data = data.get('sensor_data', [])
        
        if not shot_name or not sensor_data:
            return jsonify({"error": "Shot name and sensor data are required"}), 400
            
        # Validate the sensor data using a simplified check
        if len(sensor_data) < 3:  # At least 3 data points required
            print(f"Shot rejected: {shot_name} - too few data points")
            return jsonify({
                "message": "Data rejected - not enough data points", 
                "status": "rejected"
            }), 200
        
        # Calculate max acceleration magnitude to validate shot
        max_magnitude = 0
        for point in sensor_data:
            if all(k in point for k in ['ax', 'ay', 'az']):
                ax, ay, az = point.get('ax', 0), point.get('ay', 0), point.get('az', 0)
                magnitude = (ax**2 + ay**2 + az**2)**0.5
                max_magnitude = max(max_magnitude, magnitude)
        
        if max_magnitude < 0.2:  # Minimum threshold check
            print(f"Shot rejected: {shot_name} - insufficient acceleration")
            return jsonify({
                "message": "Shot rejected - insufficient movement detected", 
                "status": "rejected"
            }), 200
        
        # Process valid data points    
        valid_data_points = []
        for point in sensor_data:
            if all(k in point for k in ['ax', 'ay', 'az', 'gx', 'gy', 'gz']):
                valid_data_points.append({
                    'shot_name': shot_name.strip(),
                    'ax': point.get('ax', 0),
                    'ay': point.get('ay', 0),
                    'az': point.get('az', 0),
                    'gx': point.get('gx', 0),
                    'gy': point.get('gy', 0),
                    'gz': point.get('gz', 0)
                })
                
        if not valid_data_points:
            return jsonify({"error": "No valid sensor data points found"}), 400
            
        # Create DataFrame
        df = pd.DataFrame(valid_data_points)
        
        # Check if file exists and standardize existing columns if needed
        file_exists = os.path.isfile(personal_data_path)
        if file_exists:
            # Read existing data
            try:
                existing_data = pd.read_csv(personal_data_path)
                
                # Standardize columns: Convert 'ratings' to 'rating' if it exists
                if 'ratings' in existing_data.columns and 'rating' not in existing_data.columns:
                    existing_data.rename(columns={'ratings': 'rating'}, inplace=True)
                    # Save the standardized data back
                    existing_data.to_csv(personal_data_path, index=False)
                    print("Standardized CSV: 'ratings' column renamed to 'rating'")
            except Exception as e:
                print(f"Warning: Error reading existing CSV file: {str(e)}")
                file_exists = False  # Treat as if file doesn't exist
        
        # Append to CSV
        df.to_csv(personal_data_path, mode='a', header=not file_exists, index=False)
        print(f"Successfully saved {len(valid_data_points)} data points for '{shot_name}' to {personal_data_path}")
        
        return jsonify({
            "message": f"Recorded {len(valid_data_points)} data points for {shot_name}",
            "points_saved": len(valid_data_points),
            "status": "success"
        }), 200
        
    except Exception as e:
        error_msg = str(e)
        print(f"Recording error: {error_msg}")
        return jsonify({
            "error": error_msg, 
            "status": "error"
        }), 500

@app.route("/end_session", methods=["POST"])
def end_session():
    """End the recording session"""
    try:
        data = request.get_json()
        shot_name = data.get('shot_name', 'Unknown')
        
        # No actual state to clear since we're saving directly to CSV
        # This is just to confirm the session ended
        
        return jsonify({
            "message": f"Session ended for {shot_name}",
            "status": "idle"
        }), 200
        
    except Exception as e:
        print(f"End session error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/record_test_data", methods=["POST"])
def record_test_data():
    """Record test sensor data to session.json for prediction"""
    try:
        data = request.get_json()
        sensor_data = data.get('sensor_data', [])
        
        if not sensor_data:
            return jsonify({"error": "Sensor data is required"}), 400
            
        # Validate the sensor data
        if len(sensor_data) < 3:  # At least 3 data points required
            return jsonify({
                "message": "Data rejected - not enough data points", 
                "status": "rejected"
            }), 200
            
        # Create session data structure
        session_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "features": sensor_data
        }
        
        # Save to session.json
        with open('session.json', 'w') as f:
            json.dump(session_data, f, indent=2)
            
        print(f"Successfully saved {len(sensor_data)} test data points to session.json")
        
        return jsonify({
            "message": f"Recorded {len(sensor_data)} test data points",
            "points_saved": len(sensor_data),
            "status": "success"
        }), 200
        
    except Exception as e:
        error_msg = str(e)
        print(f"Test recording error: {error_msg}")
        return jsonify({
            "error": error_msg, 
            "status": "error"
        }), 500

@app.route("/convert-to-tflite", methods=["POST"])
def convert_to_tflite():
    """Convert the trained model to TFLite format as a separate step"""
    try:
        # Check if the model exists
        model_path = os.path.join(base_path, 'personal_model.h5')
        if not os.path.exists(model_path):
            return jsonify({"error": "No trained model found. Train a model first."}), 400
            
        # Check if the scaler exists
        scaler_path = os.path.join(base_path, 'personal_scaler.pkl')
        if not os.path.exists(scaler_path):
            return jsonify({"error": "Model scaler not found. Train the model again."}), 400
            
        # Load the model
        try:
            model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return jsonify({"error": f"Failed to load model: {str(e)}"}), 500
            
        # Load the data for quantization representation
        try:
            # Load data for representative dataset
            data = pd.read_csv(personal_data_path)
            
            # Standardize rating column names
            if 'ratings' in data.columns and 'rating' not in data.columns:
                data.rename(columns={'ratings': 'rating'}, inplace=True)
                
            # Get feature columns
            feature_cols = [col for col in data.columns if col != 'shot_name' and col != 'rating']
            X = data[feature_cols]
            
            # Load the scaler
            scaler = joblib.load(scaler_path)
            X_scaled = scaler.transform(X)
            
            # Create representative dataset
            quantize_data = []
            for i in range(min(10, len(X_scaled))):
                quantize_data.append(X_scaled[i])
                
            print(f"Created representative dataset with {len(quantize_data)} samples")
        except Exception as e:
            print(f"Error preparing quantization data: {str(e)}")
            # Continue without quantization data
            quantize_data = None
            
        # Convert to TFLite
        try:
            tflite_path = model_path.replace('.h5', '.tflite')
            
            # Simple conversion without quantization if data preparation failed
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            
            # Only use quantization if we have data
            if quantize_data:
                def representative_dataset_generator():
                    for value in quantize_data:
                        # Each scalar value must be inside of a 2D array that is wrapped in a list
                        yield [np.array(value, dtype=np.float32, ndmin=2)]
                
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.representative_dataset = representative_dataset_generator
                print("Using quantization optimization")
            
            # Convert the model
            tflite_model = converter.convert()
            
            # Save the TFLite model
            with tf.io.gfile.GFile(tflite_path, 'wb') as f:
                f.write(tflite_model)
                
            print(f"TFLite model saved to {tflite_path}")
            
            # Try to create C header
            try:
                c_header_dir = os.path.join(base_path, 'c_header')
                if not os.path.exists(c_header_dir):
                    os.makedirs(c_header_dir)
                
                c_header_path = os.path.join(c_header_dir, 'model.h')
                tflite_to_c(tflite_path, c_header_dir)
                print(f"C header saved to {c_header_path}")
                
                c_header_status = "C header generated successfully"
            except Exception as e:
                print(f"C header generation error: {str(e)}")
                c_header_status = f"TFLite model created, but C header generation failed: {str(e)}"
            
            return jsonify({
                "message": "TFLite model created successfully!",
                "c_header_status": c_header_status,
                "tflite_path": tflite_path
            }), 200
            
        except Exception as e:
            print(f"TFLite conversion error: {str(e)}")
            traceback.print_exc()
            return jsonify({"error": f"TFLite conversion failed: {str(e)}"}), 500
            
    except Exception as e:
        print(f"TFLite process error: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)