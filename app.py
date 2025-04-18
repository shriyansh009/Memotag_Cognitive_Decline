from flask import Flask, render_template, request, redirect, url_for ,jsonify
import os
from werkzeug.utils import secure_filename
import uuid
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from utils.features import compute_features, summarize_text, find_hesitations
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# For demonstration: collect all uploaded features for clustering
feature_collection = []

def calculate_risk(features):
    selected_features = ["fillers_per_100_words", "lexical_diversity", "pause_estimate", 
                         "pitch_variability", "speech_rate", "energy_rms", "tempo"]
    
    # Store for batch clustering (simulating anomaly detection)
    feature_collection.append({k: features[k] for k in selected_features})
    df = pd.DataFrame(feature_collection)

    # Only cluster if we have more than 2 samples
    if len(df) >= 3:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)

        model = KMeans(n_clusters=2, random_state=42)
        labels = model.fit_predict(X_scaled)

        input_cluster = labels[-1]
        cluster_sizes = pd.Series(labels).value_counts()

        # Simple anomaly detection logic: smaller cluster is "High Risk"
        risky_cluster = cluster_sizes.idxmin()
        level = "High" if input_cluster == risky_cluster else "Low"
        score = int(input_cluster == risky_cluster) * 2  # score: 0 or 2
    else:
        # fallback scoring if not enough data
        score = 0
        if features.get("fillers_per_100_words", 0) > 5: score += 1
        if features.get("lexical_diversity", 1) < 0.5: score += 1
        if features.get("pause_estimate", 0) > 7: score += 1
        if features.get("pitch_variability", 15) < 10: score += 1
        level = ["Low", "Moderate", "High"][min(score, 2)]

    return score, level

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file and file.filename.endswith('.wav'):
        filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return redirect(url_for('result', filename=filename))
    return "Invalid file", 400

@app.route('/result', methods=['GET', 'POST'])
def result():
    features = None
    risk_score = None
    risk_level = None
    summary = ""
    hesitations = []
    transcript = ""
    
    filename = request.args.get('filename')
    if not filename:
        return "No file provided", 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    features, transcript = compute_features(filepath, return_text=True)
    summary = summarize_text(transcript)
    hesitations = find_hesitations(transcript)
    risk_score, risk_level = calculate_risk(features)

    return render_template('result.html',
        features=features,
        risk_score=risk_score,
        risk_level=risk_level,
        summary=summary,
        hesitations=hesitations,
        transcript=transcript
    )

@app.route('/api/analyze', methods=['POST'])
def analyze_api():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not file or not file.filename.endswith('.wav'):
        return jsonify({'error': 'Invalid file format, only .wav supported'}), 400

    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        features, transcript = compute_features(filepath, return_text=True)
        summary = summarize_text(transcript)
        hesitations = find_hesitations(transcript)
        risk_score, risk_level = calculate_risk(features)

        def convert_np(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_np(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_np(i) for i in obj]
            return obj


        return jsonify({
            'filename': filename,
            'transcript': transcript,
            'summary': summary,
            'hesitations': hesitations,
            'features': convert_np(features),
            'risk_score': risk_score,
            'risk_level': risk_level
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500    

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
