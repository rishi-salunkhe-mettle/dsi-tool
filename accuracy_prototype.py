import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc, brier_score_loss
import matplotlib.pyplot as plt

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
        'AUROC': auroc,
        'ROC Curve': (fpr_vals, tpr_vals)
    }

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

def plot_roc_curve(fpr, tpr):
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    st.pyplot(plt)

st.title("Machine Learning Model Accuracy Metrics")

uploaded_file = st.file_uploader("Upload Data (CSV)", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    
    st.subheader("First 10 Rows of Data")
    st.write(data.head(10))
    
    y_true = data.iloc[:, -2].values  # Assuming second last column is the actual label
    y_pred = data.iloc[:, -1].values  # Assuming last column is the predictions
    
    metrics = calculate_metrics(y_true, y_pred)
    
    st.subheader("Accuracy Metrics")
    for key, value in metrics.items():
        if key != 'ROC Curve':
            st.write(f"{key}: {value:.4f}")
    
    st.subheader("ROC Curve")
    plot_roc_curve(*metrics['ROC Curve'])
    
    selected_feature = st.selectbox("Select an input column for subgroup analysis", data.columns[:-2])
    subgroup_metrics_df = calculate_subgroup_metrics(data, y_true, y_pred, selected_feature)
    
    st.subheader(f"Subgroup Metrics for {selected_feature}")
    st.dataframe(subgroup_metrics_df)
