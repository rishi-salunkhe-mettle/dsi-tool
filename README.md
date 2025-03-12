# Machine Learning Model Accuracy Metrics

This Streamlit application calculates and displays accuracy metrics for a machine learning model. It supports subgroup analysis and visualizes ROC curves.

## Installation

1. Clone this repository:
   ```sh
   git clone <repository_url>
   cd <repository_name>
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
   streamlit run Streamlit_ML_Metrics.py
   ```

2. Upload a CSV file containing input features along with training labels (second last column) and prediction labels (last column).

3. The application will display:
   - The first 10 rows of the uploaded dataset
   - Accuracy metrics (TP, TN, FP, FN, Precision, Recall, F1 Score, Brier Score, AUROC, etc.)
   - ROC Curve visualization
   - Subgroup analysis based on the selected input feature

## CSV File Format
The uploaded CSV should have the following structure:

| Age | Gender | Race | Blood Sugar Level | Family Diabetes History | Training Label | Prediction Label |
|-----|--------|------|------------------|----------------------|---------------|-----------------|
| 45  | Male   | White | 120              | Yes                  | 1             | 1               |
| 32  | Female | Asian | 85               | No                   | 0             | 0               |
| ... | ...    | ...  | ...              | ...                  | ...           | ...             |

## Dependencies
- Python 3.x
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## License
This project is licensed under the MIT License.

