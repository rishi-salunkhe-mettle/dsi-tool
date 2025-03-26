# DSI Tool

This tool calculates and displays accuracy metrics for a healthcare machine learning model. It supports subgroup analysis and visualizes ROC curves.

## Installation

1. Clone this repository:
   ```sh
   git clone https://github.com/rishi-salunkhe-mettle/dsi-tool
   cd dsi-tool
   ```

2. Create a virtual environment (optional but recommended):
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Execution

1. Run the Streamlit application:
   ```sh
   streamlit run accuracy_prototype.py
   ```

2. Upload a CSV file (sample file: diabetes_prediction_data.csv) containing input features along with training labels (second last column) and prediction labels (last column).

3. The application will display:
   - The first 10 rows of the uploaded dataset
   - Accuracy metrics (TP, TN, FP, FN, Precision, Recall, F1 Score, Brier Score, AUROC, etc.)
   - ROC Curve visualization
   - Subgroup analysis based on the selected input feature

## CSV File Format
The uploaded CSV should have the following structure:

 | Patient_ID | Prediction_Timestamp | Predicted_Probability | Age | Gender | Race  | HbA1c (%) | eGFR | UACR | Comorbidities        | Actual_Outcome | Predicted_Outcome | 
 |------------|----------------------|-----------------------|-----|--------|-------|-----------|------|------|----------------------|----------------|-------------------|
 | P001       | 2024-03-15 08:00 AM  | 0.37                  | 64  | Female | Black | 6.6       | 35   | 634  | "CKD, Heart Failure" | 0              | 0                 | 
 | P002       | 2024-03-15 08:15 AM  | 0.95                  | 29  | Female | White | 10.4      | 118  | 22   | Heart Failure        | 0              | 1                 | 
 | ...        | ...                  | ...                   | ... | ...    | ...   | ...       | ...  | ...  | ...                  | ...            | ...               | 

## Dependencies
- Python 3.x
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## License
This project is licensed under the MIT License.

