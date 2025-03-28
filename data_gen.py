import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# Generate 1000 rows of synthetic data
num_rows = 1000

data = {
    "Patient_ID": [f"P{str(i).zfill(3)}" for i in range(1, num_rows + 1)],
    "Prediction_Timestamp": [(datetime(2024, 3, 15, 8, 0) + timedelta(minutes=15 * i)).strftime("%Y-%m-%d %I:%M %p") for i in range(num_rows)],
    "Predicted_Probability": np.round(np.random.uniform(0, 1, num_rows), 2),
    "Age": np.random.randint(18, 91, num_rows),
    "Gender": np.random.choice(["Male", "Female"], num_rows),
    "Race": np.random.choice(["White", "Black", "Asian", "Hispanic", "Other"], num_rows),
    "HbA1c (%)": np.round(np.random.uniform(4.5, 12.0, num_rows), 1),
    "eGFR": np.random.randint(15, 120, num_rows),
    "UACR": np.random.randint(10, 1000, num_rows),
    "Comorbidities": np.random.choice([
        "Diabetes", "Hypertension", "CKD", "Heart Failure", "Diabetes, Hypertension", "Diabetes, CKD", "CKD, Heart Failure", "None"
    ], num_rows),
    "Actual_Outcome": np.random.choice([0, 1], num_rows),
    "Predicted_Outcome": np.random.choice([0, 1], num_rows)
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("diabetes_prediction_data.csv", index=False)

print("CSV file 'diabetes_prediction_data.csv' created successfully!")
