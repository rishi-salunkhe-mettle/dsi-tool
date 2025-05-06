from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc, brier_score_loss
import psycopg2
from psycopg2.extras import execute_values

app = Flask(__name__)
CORS(app)

ACC_FILE = "accuracy_history.csv"

# PostgreSQL connection config
DB_CONFIG = {
    'dbname': 'lava',
    'user': 'lava',
    'password': 'password',
    'host': 'localhost',
    'port': 5433
}

def insert_into_postgres(data):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        insert_query = """
            INSERT INTO predictions (patient_id, prediction_timestamp, predicted_prob, predicted_outcome)
            VALUES %s
        """
        execute_values(cursor, insert_query, data)
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print("Database error:", e)
        return False

def calculate_metrics(y_true, y_pred):
    for i in range(len(y_pred)):
        y_pred[i] = (1 - y_true[i]) if y_pred[i] == 9 else y_pred[i]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = cm[0, 0] if cm.shape[0] > 0 else 0
        fp = cm[0, 1] if cm.shape[1] > 1 else 0
        fn = cm[1, 0] if cm.shape[0] > 1 else 0
        tp = cm[1, 1] if cm.shape[1] > 1 else 0

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    brier = brier_score_loss(y_true, y_pred)

    if len(set(y_true)) > 1:
        fpr_vals, tpr_vals, _ = roc_curve(y_true, y_pred)
        auroc = auc(fpr_vals, tpr_vals)
    else:
        fpr_vals, tpr_vals, auroc = [0], [0], 0

    return {
        'Accuracy': (tp + tn) / (tp + tn + fp + fn),
        'True Positive': tp,
        'True Negative': tn,
        'False Positive': fp,
        'False Negative': fn,
        'True Positive Rate': tpr,
        'True Negative Rate': tnr,
        'False Positive Rate': fpr,
        'False Negative Rate': fnr,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Brier Score': brier,
        'AUROC': auroc
    }

def save_accuracy(metrics):
    df = pd.DataFrame([metrics])
    df['Date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        df_existing = pd.read_csv(ACC_FILE)
        df_existing = pd.concat([df_existing, df], ignore_index=True)
    except FileNotFoundError:
        df_existing = df
    df_existing.to_csv(ACC_FILE, index=False)

def delete_accuracy_history():
    if os.path.exists(ACC_FILE):
        os.remove(ACC_FILE)
        return True
    return False

def calculate_subgroup_metrics(data, y_true, y_pred, selected_feature):
    subgroup_metrics = []
    unique_values = data[selected_feature].unique()
    for value in unique_values:
        mask = data[selected_feature] == value
        if np.sum(mask) > 0:
            metrics = calculate_metrics(y_true[mask], y_pred[mask])
            metrics = {'Subgroup': value, **metrics}
            subgroup_metrics.append(metrics)
    return pd.DataFrame(subgroup_metrics)

@app.route('/calculate_metrics', methods=['GET'])
def metrics_endpoint():
    prediction_type = request.args.get('prediction_type')
    if not prediction_type:
            return jsonify({'error': 'No prediction_type provided'}), 400

    DATA_FILE = 'diabetes_prediction_' + prediction_type + '_data.csv'
    data = pd.read_csv(DATA_FILE)

    # Ensure Prediction_Timestamp is datetime
    data['Prediction_Timestamp'] = pd.to_datetime(data['Prediction_Timestamp'])

    # Get optional date filters from query params
    start_date_str = request.args.get('start_date')
    end_date_str = request.args.get('end_date')

    # Filter by date range if provided
    if start_date_str and end_date_str:
        try:
            start_date = pd.to_datetime(start_date_str)
            end_date = pd.to_datetime(end_date_str)
            data = data[(data['Prediction_Timestamp'] >= start_date) & (data['Prediction_Timestamp'] <= end_date)]
        except Exception as e:
            return jsonify({'error': f'Invalid date format: {e}'}), 400

    # Sort and keep latest per patient
    data = data.sort_values('Prediction_Timestamp').drop_duplicates('Patient_ID', keep='last')

    if data.empty:
        return jsonify({'error': 'No data available for the given date range'}), 400

    # Drop identifiers
    data = data.drop(columns=['Patient_ID', 'Prediction_Timestamp'])

    y_true = data.iloc[:, -2].values
    y_pred = data.iloc[:, -1].values

    if isinstance(y_pred[0], str):
        data = request.get_json()
        if not data or 'categories' not in data:
            return jsonify({'error': 'categories is required in JSON body'}), 400
        
        categories = data['categories']

        # Create mapping from categories to values between 0 and 1
        mapped_values = np.linspace(0, 1, len(categories))
        category_to_value = dict(zip(categories, mapped_values))

        # Replace values in y_pred with corresponding mapped values
        y_pred = [category_to_value[val] for val in y_pred]
    
    threshold = request.args.get('threshold')  # Use query parameter
    
    if y_pred[0] not in [0, 1]:
        if not threshold:
            return jsonify({'error': 'No threshold provided'}), 400

        threshold = float(threshold)

        def modify_list(values, threshold):
            updated_values = []
            for val in values:
                if abs(val - 0) <= threshold:
                    updated_values.append(0)
                elif abs(val - 1) <= threshold:
                    updated_values.append(1)
                else:
                    updated_values.append(9)
            return updated_values
        
        y_pred = modify_list(y_pred, threshold)

    metrics = calculate_metrics(y_true, y_pred)
    save_accuracy(metrics)

    # Convert numpy types to native Python types
    metrics = {k: float(v) if isinstance(v, (np.float32, np.float64)) else int(v) if isinstance(v, (np.int32, np.int64)) else v for k, v in metrics.items()}
    return jsonify(metrics), 200

@app.route('/calculate_metrics', methods=['POST'])
def post_metrics_endpoint():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    data = pd.read_csv(file)
    data = data.drop(columns=['Patient_ID', 'Prediction_Timestamp'])
    y_true = data.iloc[:, -2].values
    y_pred = data.iloc[:, -1].values
    metrics = calculate_metrics(y_true, y_pred)
    save_accuracy(metrics)

    metrics = {k: float(v) if isinstance(v, (np.float32, np.float64)) else int(v) if isinstance(v, (np.int32, np.int64)) else v for k, v in metrics.items()}
    return jsonify(metrics), 200

@app.route('/subgroup_metrics', methods=['GET'])
def subgroup_metrics_endpoint():
    selected_feature = request.args.get('feature')  # Use query parameter
    if not selected_feature:
        return jsonify({'error': 'No feature provided'}), 400

    if selected_feature not in ['Gender', 'Race', 'Comorbidities']:
        return jsonify({'error': 'Provided feature is not supported. Please provide one of the following: Age, Gender, Race, Comorbidities'}), 400

    
    prediction_type = request.args.get('prediction_type')  # Use query parameter
    if not prediction_type:
        return jsonify({'error': 'No prediction_type provided'}), 400
    
    DATA_FILE = 'diabetes_prediction_' + prediction_type + '_data.csv'
    data = pd.read_csv(DATA_FILE)

    data = data.drop(columns=['Patient_ID', 'Prediction_Timestamp'])
    y_true = data.iloc[:, -2].values
    y_pred = data.iloc[:, -1].values

    subgroup_df = calculate_subgroup_metrics(data, y_true, y_pred, selected_feature)
    return jsonify(subgroup_df.to_dict(orient='records'))


@app.route('/clear_history', methods=['DELETE'])
def clear_history():
    if delete_accuracy_history():
        return jsonify({'message': 'Accuracy history deleted'}), 200
    return jsonify({'message': 'No history to delete'}), 200

@app.route('/upload_prediction_data', methods=['POST'])
def upload_prediction_data():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'File must be a CSV'}), 400

    df = pd.read_csv(file)
    required_cols = ['Patient_ID', 'Prediction_Timestamp', 'Predicted_Probability', 'Predicted_Outcome']
    if not all(col in df.columns for col in required_cols):
        return jsonify({'error': 'CSV missing required columns'}), 400

    rows = df[required_cols].values.tolist()

    success = insert_into_postgres(rows)
    if success:
        return jsonify({'message': 'Data inserted into PostgreSQL successfully'}), 200
    else:
        return jsonify({'error': 'Failed to insert into database'}), 500

@app.route('/get_prediction_by_patient', methods=['POST'])
def get_prediction_by_patient():
    data = request.get_json()
    if not data or 'patient_id' not in data:
        return jsonify({'error': 'patient_id is required in JSON body'}), 400

    patient_id = data['patient_id']

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        query = """
            SELECT patient_id, prediction_timestamp, predicted_prob, predicted_outcome
            FROM predictions
            WHERE patient_id = %s
        """
        cursor.execute(query, (patient_id,))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        if not rows:
            return jsonify({'message': f'No data found for patient_id: {patient_id}'}), 404

        result = [{
            'patient_id': row[0],
            'prediction_timestamp': row[1].isoformat() if hasattr(row[1], 'isoformat') else str(row[1]),
            'predicted_prob': float(row[2]),
            'predicted_outcome': int(row[3])
        } for row in rows]

        return jsonify(result), 200

    except Exception as e:
        print("Database fetch error:", e)
        return jsonify({'error': 'Failed to fetch data from database'}), 500

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0", port=5000)
