from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc, brier_score_loss

app = Flask(__name__)
CORS(app)

ACC_FILE = "accuracy_history.csv"

def calculate_metrics(y_true, y_pred):
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

@app.route('/calculate_metrics', methods=['POST'])
def metrics_endpoint():
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

@app.route('/subgroup_metrics', methods=['POST'])
def subgroup_metrics_endpoint():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    
    selected_feature = request.args.get('feature')  # Use query parameter
    if not selected_feature:
        return jsonify({'error': 'No feature provided'}), 400

    if selected_feature not in ['Age', 'Gender', 'Race', 'Comorbidities']:
        return jsonify({'error': 'Provided feature is not supported. Please provide one of the following: Age, Gender, Race, Comorbidities'}), 400

    data = pd.read_csv(file)
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

if __name__ == '__main__':
    app.run(debug=True)
