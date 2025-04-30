import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Parameters
patients = 200
rows_per_patient = 5
total_rows = patients * rows_per_patient
base_time = datetime(2024, 3, 15, 8, 0)

# Health trajectory types
trajectories = ['improving', 'worsening', 'stable']

data = []

for i in range(1, patients + 1):
    patient_id = f"P{str(i).zfill(3)}"
    trajectory = np.random.choice(trajectories)
    age = np.random.randint(18, 91)
    birthdate = (base_time - timedelta(days=age * 365 + np.random.randint(0, 365))).date()
    gender = np.random.choice(["Male", "Female"])
    race = np.random.choice(["White", "Black", "Asian", "Hispanic", "Other"])
    comorbidity = np.random.choice([
        "Diabetes", "Hypertension", "Heart Failure",
        "Diabetes, Hypertension", "Hypertension, Heart Failure", "Diabetes, Heart Failure", 
        "Diabetes, Hypertension, Heart Failure", "None"
    ])

    # Starting values
    hba1c = np.round(np.random.uniform(5.5, 10.0), 1)
    egfr = np.random.randint(30, 90)
    uacr = np.random.randint(30, 500)

    actual_outcome = np.random.choice([0, 1])
    predicted_outcome = np.random.choice([0, 1])

    for j in range(rows_per_patient):
        timestamp = base_time + timedelta(days=j*30, minutes=15*i)  # roughly monthly intervals

        # Apply change based on trajectory
        if trajectory == 'improving':
            hba1c = max(4.5, hba1c - np.random.uniform(0.1, 0.5))
            egfr = min(120, egfr + np.random.uniform(1, 5))
            uacr = max(10, uacr - np.random.uniform(5, 30))
        elif trajectory == 'worsening':
            hba1c = min(12.0, hba1c + np.random.uniform(0.1, 0.5))
            egfr = max(15, egfr - np.random.uniform(1, 5))
            uacr = min(1000, uacr + np.random.uniform(5, 30))
        elif trajectory == 'stable':
            hba1c = np.clip(hba1c + np.random.uniform(-0.1, 0.1), 5.5, 6.5)
            egfr = np.clip(egfr + np.random.uniform(-1, 1), 90, 120)
            uacr = np.clip(uacr + np.random.uniform(-5, 5), 10, 30)

        # Match predicted_probability with predicted_outcome
        if predicted_outcome == 0:
            predicted_probability = np.round(np.random.uniform(0.0, 0.49), 2)
        else:
            predicted_probability = np.round(np.random.uniform(0.5, 1.0), 2)

        data.append({
            "Patient_ID": patient_id,
            "Prediction_Timestamp": timestamp.strftime("%Y-%m-%d %I:%M %p"),
            "Predicted_Probability": predicted_probability,
            "Birthdate": birthdate.strftime("%Y-%m-%d"),
            "Gender": gender,
            "Race": race,
            "HbA1c (%)": np.round(hba1c, 1),
            "eGFR": int(round(egfr)),
            "UACR": int(round(uacr)),
            "Comorbidities": comorbidity,
            "Actual_Outcome": actual_outcome,
            "Predicted_Outcome": predicted_outcome
        })

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("diabetes_prediction_trajectory_data.csv", index=False)

print("CSV file 'diabetes_prediction_trajectory_data.csv' created successfully!")
