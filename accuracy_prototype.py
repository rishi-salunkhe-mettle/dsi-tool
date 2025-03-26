import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc, brier_score_loss
import matplotlib.pyplot as plt
import seaborn as sns

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
        'ROC Curve': (fpr_vals, tpr_vals),
        'Confusion Matrix': cm
    }

def calculate_subgroup_metrics(data, y_true, y_pred, selected_feature):
    subgroup_metrics = []
    unique_values = data[selected_feature].unique()
    for value in unique_values:
        mask = data[selected_feature] == value
        if np.sum(mask) > 0:
            metrics = calculate_metrics(y_true[mask], y_pred[mask])
            metrics['Confusion Matrix'] = metrics['Confusion Matrix'].tolist()
            metrics = {'Subgroup': value, **metrics}
            subgroup_metrics.append(metrics)
    return pd.DataFrame(subgroup_metrics)

def plot_subgroup_metrics(subgroup_metrics_df, selected_feature):
    subgroup_metrics_df = subgroup_metrics_df.drop(columns=['True Positive', 'True Negative', 'False Positive', 'False Negative', 'ROC Curve', 'Confusion Matrix'])
    melted_df = subgroup_metrics_df.melt(id_vars=['Subgroup'], var_name='Metric', value_name='Score')
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=melted_df, x='Metric', y='Score', hue='Subgroup', palette='tab10')

    # Add value labels on top of bars
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f'{p.get_height():.2f}', 
                (p.get_x() + p.get_width() / 2, p.get_height()), 
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Score')
    plt.xlabel('Accuracy Metrics')
    plt.title(f'Subgroup Analysis across {selected_feature}')
    plt.ylim(0, 1.2)
    plt.legend(loc='upper right')
    st.pyplot(plt)
    st.markdown("This bar chart visualizes multiple accuracy metrics across different subgroups within the selected input column.")

def plot_roc_curve(fpr, tpr):
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    st.pyplot(plt)
    st.markdown("**ROC Curve**: plots the True Positive Rate against the False Positive Rate, showing the performance of the classification model at different thresholds.")
    st.markdown("**AUROC (Area Under the Receiver Operating Characteristic Curve)**: measures a classifier's ability to distinguish between classes, with a higher value (closer to 1) indicating better performance.")

def plot_confusion_matrix(cm, tpr, tnr, fpr, fnr):
    plt.figure(figsize=(6, 5))
    labels = [[f'TP: {cm[1,1]}\nTPR: {tpr:.2f}', f'FN: {cm[1,0]}\nFNR: {fnr:.2f}'],
              [f'FP: {cm[0,1]}\nFPR: {fpr:.2f}', f'TN: {cm[0,0]}\nTNR: {tnr:.2f}']]
    ax = sns.heatmap([[cm[1,1], cm[1,0]], [cm[0,1], cm[0,0]]], annot=labels, fmt='', cmap='Blues', xticklabels=['Pred 1', 'Pred 0'], yticklabels=['True 1', 'True 0'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    st.pyplot(plt)
    st.markdown("**True Positive Rate (TPR)**: The proportion of actual positives correctly identified as positive (also called sensitivity or recall).")
    st.markdown("**True Negative Rate (TNR)**: The proportion of actual negatives correctly identified as negative (also called specificity).")
    st.markdown("**False Positive Rate (FPR)**: The proportion of actual negatives incorrectly classified as positive.")
    st.markdown("**False Negative Rate (FNR)**: The proportion of actual positives incorrectly classified as negative.")

def plot_metric_bar_chart(metrics):
    plt.figure(figsize=(6, 4))
    metric_names = ['Precision', 'Recall', 'F1 Score', 'Brier Score']
    metric_values = [metrics[m] for m in metric_names]
    ax = sns.barplot(x=metric_names, y=metric_values, palette='Blues')
    for i, v in enumerate(metric_values):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=10, fontweight='bold')
    plt.ylim(0, 1)
    plt.ylabel('Score')
    st.pyplot(plt)
    st.markdown("**Precision**: The proportion of predicted positives that are actually positive.")
    st.markdown("**Recall**: The proportion of actual positives that are correctly identified (same as TPR).")
    st.markdown("**F1 Score**: The harmonic mean of precision and recall, providing a balance between the two.")
    st.markdown("**Brier Score**: A measure of how well probabilistic predictions are calibrated, with lower values indicating better accuracy.")

st.title("Onset of CKD")

uploaded_file = st.file_uploader("Upload Data (CSV)", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    
    st.subheader("First 10 Rows of Data")
    st.write(data.head(10))
    
    data = data.drop(columns=['Patient_ID', 'Prediction_Timestamp'])

    y_true = data.iloc[:, -2].values  # Assuming second last column is the actual label
    y_pred = data.iloc[:, -1].values  # Assuming last column is the predictions
    
    metrics = calculate_metrics(y_true, y_pred)
        
    st.subheader("Confusion Matrix")
    plot_confusion_matrix(metrics['Confusion Matrix'], metrics['True Positive Rate'], metrics['True Negative Rate'], metrics['False Positive Rate'], metrics['False Negative Rate'])
    
    st.subheader("Accuracy Metrics")
    plot_metric_bar_chart(metrics)

    st.subheader("ROC Curve")
    plot_roc_curve(*metrics['ROC Curve'])

    st.subheader("Subgroup Analysis")
    selected_feature = st.selectbox("Select an input column for subgroup analysis", ['Age', 'Gender', 'Race', 'Comorbidities'])
    subgroup_metrics_df = calculate_subgroup_metrics(data, y_true, y_pred, selected_feature)
    
    st.subheader(f"Subgroup Metrics for {selected_feature}")
    st.dataframe(subgroup_metrics_df)

    selected_feature = st.selectbox("Select an input column for subgroup analysis visualization", ['Gender', 'Race', 'Comorbidities'])
    subgroup_metrics_df = calculate_subgroup_metrics(data, y_true, y_pred, selected_feature)
    
    st.subheader(f"Subgroup Analysis Visualization: {selected_feature}")
    plot_subgroup_metrics(subgroup_metrics_df, selected_feature)