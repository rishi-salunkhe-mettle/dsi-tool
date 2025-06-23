import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math

# Parameters
patients = 200
rows_per_patient = 5
total_rows = patients * rows_per_patient
base_time = datetime(2024, 3, 15, 8, 0)

health_trajectories = ["improving", "worsening", "stable"]
data = []

for i in range(1, patients + 1):
    patient_id = f"P{str(i+200).zfill(3)}"
    age = np.random.randint(40, 59)
    birthdate = (base_time - timedelta(days=age * 365 + np.random.randint(0, 365))).date()
    gender = np.random.choice(["Male", "Female"])
    race = np.random.choice(["White", "African American"])
    actual_outcome = np.random.choice([0, 1])
    health_trajectory = np.random.choice(health_trajectories)

    # Generate new biometrics
    total_cholesterol = np.round(np.random.uniform(150, 280), 1)
    hdl_cholesterol = np.round(np.random.uniform(40, 100), 1)
    systolic_bp = np.random.randint(90, 180)

    # Generate categorical health statuses
    diabetes_status = np.random.choice(["Diabetic", "Non-Diabetic"])
    current_smoking_status = np.random.choice(["Smoker", "Non-Smoker"])
    hypertension_treatment_status = np.random.choice(["On Treatment", "Not on Treatment"])

    for j in range(rows_per_patient):
        timestamp = base_time + timedelta(days=j*30, minutes=15*i)  # roughly monthly intervals

        # Adjust biometrics based on health trajectory
        if health_trajectory == "improving":
            total_cholesterol = max(150, total_cholesterol - np.random.uniform(1, 5))
            hdl_cholesterol = min(100, hdl_cholesterol + np.random.uniform(0.5, 2))
            systolic_bp = max(90, systolic_bp - np.random.randint(0, 3))
        elif health_trajectory == "worsening":
            total_cholesterol = min(280, total_cholesterol + np.random.uniform(1, 5))
            hdl_cholesterol = max(40, hdl_cholesterol - np.random.uniform(0.5, 2))
            systolic_bp = min(180, systolic_bp + np.random.randint(0, 3))
        # Stable condition
        total_cholesterol += np.random.uniform(-2, 2)
        hdl_cholesterol += np.random.uniform(-1, 1)
        systolic_bp += np.random.randint(-1, 2)
        
        # Keep values within realistic bounds
        total_cholesterol = round(min(max(total_cholesterol, 150), 280), 1)
        hdl_cholesterol = round(min(max(hdl_cholesterol, 40), 100), 1)
        systolic_bp = min(max(systolic_bp, 90), 180)

        # Calculate actual outcome based on risk factors
        high_risk = (
            total_cholesterol > 240 and
            hdl_cholesterol < 40 and
            systolic_bp > 160 and
            diabetes_status == "Diabetic" and
            current_smoking_status == "Smoker" and
            hypertension_treatment_status == "Not on Treatment"
        )
        actual_outcome = 1 if high_risk and np.random.rand() < 0.6 else actual_outcome

        # Calculate additional risk metrics
        is_male = gender == "Male"
        is_aa = race == "African American"
        ln_age = math.log(age)
        ln_total_chol = math.log(total_cholesterol)
        ln_hdl = math.log(hdl_cholesterol)

        # Ten-Year ASCVD Score (simplified)
        if is_aa and not is_male:
            s010_ret = 0.95334
            mnxb_ret = 86.6081
            predict_ret = (17.1141 * ln_age) + (0.9396 * ln_total_chol) + (-18.9196 * ln_hdl)
        elif not is_aa and not is_male:
            s010_ret = 0.96652
            mnxb_ret = -29.1817
            predict_ret = (-29.799 * ln_age) + (4.884 * (ln_age ** 2)) + (13.54 * ln_total_chol) + (-13.578 * ln_hdl)
        elif is_aa and is_male:
            s010_ret = 0.89536
            mnxb_ret = 19.5425
            predict_ret = (2.469 * ln_age) + (0.302 * ln_total_chol) + (-0.307 * ln_hdl)
        else:
            s010_ret = 0.91436
            mnxb_ret = 61.1816
            predict_ret = (12.344 * ln_age) + (11.853 * ln_total_chol) + (-7.99 * ln_hdl)
        ten_year_score = round((1 - (s010_ret ** math.exp(predict_ret - mnxb_ret))) * 100, 1)

        # Lifetime ASCVD Risk
        if 20 <= age <= 59:
            major = (1 if total_cholesterol >= 240 else 0) + \
                    (1 if systolic_bp >= 160 or hypertension_treatment_status == "On Treatment" else 0) + \
                    (1 if current_smoking_status == "Smoker" else 0) + \
                    (1 if diabetes_status == "Diabetic" else 0)
            elevated = (1 if (200 <= total_cholesterol < 240 or \
                             (140 <= systolic_bp < 160 and hypertension_treatment_status == "Not on Treatment")) and major == 0 else 0)
            all_optimal = (1 if (total_cholesterol < 180 and systolic_bp < 120 and hypertension_treatment_status == "Not on Treatment") and major == 0 else 0)
            not_optimal = (1 if (180 <= total_cholesterol < 200 or \
                                 (120 <= systolic_bp < 140 and hypertension_treatment_status == "Not on Treatment")) and elevated == 0 and major == 0 else 0)
            params = {
                "Male": {"major2": 69, "major1": 50, "elevated": 46, "notOptimal": 36, "allOptimal": 5},
                "Female": {"major2": 50, "major1": 39, "elevated": 39, "notOptimal": 27, "allOptimal": 8}
            }
            if major > 1:
                lifetime_risk = params[gender]["major2"]
            elif major == 1:
                lifetime_risk = params[gender]["major1"]
            elif elevated == 1:
                lifetime_risk = params[gender]["elevated"]
            elif not_optimal == 1:
                lifetime_risk = params[gender]["notOptimal"]
            elif all_optimal == 1:
                lifetime_risk = params[gender]["allOptimal"]
            else:
                lifetime_risk = 0
        else:
            lifetime_risk = np.nan

        # Lowest Risk Calculations
        if is_aa and not is_male:
            s010_ret = 0.95334
            mnxb_ret = 86.6081
            predict_ret = (17.1141 * ln_age) + (0.9396 * math.log(130)) + (-18.9196 * math.log(100))
        elif not is_aa and not is_male:
            s010_ret = 0.96652
            mnxb_ret = -29.1817
            predict_ret = (-29.799 * ln_age) + (4.884 * (ln_age ** 2)) + (13.54 * math.log(130)) + (-13.578 * math.log(100))
        elif is_aa and is_male:
            s010_ret = 0.89536
            mnxb_ret = 19.5425
            predict_ret = (2.469 * ln_age) + (0.302 * math.log(130)) + (-0.307 * math.log(100))
        else:
            s010_ret = 0.91436
            mnxb_ret = 61.1816
            predict_ret = (12.344 * ln_age) + (11.853 * math.log(130)) + (-7.99 * math.log(100))
        lowest_ten_year = round((1 - (s010_ret ** math.exp(predict_ret - mnxb_ret))) * 100, 1)

        if 20 <= age <= 59:
            major = (1 if 130 >= 240 else 0) + \
                    (1 if 90 >= 160 or "Not on Treatment" == "On Treatment" else 0) + \
                    (1 if "Non-Smoker" == "Smoker" else 0) + \
                    (1 if "Non-Diabetes" == "Diabetic" else 0)
            elevated = (1 if (200 <= 130 < 240 or \
                             (140 <= 90 < 160 and "Not on Treatment" == "Not on Treatment")) and major == 0 else 0)
            all_optimal = (1 if (130 < 180 and 90 < 120 and "Not on Treatment" == "Not on Treatment") and major == 0 else 0)
            not_optimal = (1 if (180 <= 130 < 200 or \
                                 (120 <= 90 < 140 and "Not on Treatment" == "Not on Treatment")) and elevated == 0 and major == 0 else 0)
            params = {
                "Male": {"major2": 69, "major1": 50, "elevated": 46, "notOptimal": 36, "allOptimal": 5},
                "Female": {"major2": 50, "major1": 39, "elevated": 39, "notOptimal": 27, "allOptimal": 8}
            }
            if major > 1:
                lowest_lifetime = params[gender]["major2"]
            elif major == 1:
                lowest_lifetime = params[gender]["major1"]
            elif elevated == 1:
                lowest_lifetime = params[gender]["elevated"]
            elif not_optimal == 1:
                lowest_lifetime = params[gender]["notOptimal"]
            elif all_optimal == 1:
                lowest_lifetime = params[gender]["allOptimal"]
            else:
                lowest_lifetime = 0
        else:
            lowest_lifetime = np.nan

        # Potential Risk Reduction (simplified)
        potential_risk = max(0, ten_year_score - lowest_ten_year) if not np.isnan(lowest_ten_year) else np.nan

        data.append({
            "Patient_ID": patient_id,
            "Prediction_Timestamp": timestamp.strftime("%Y-%m-%d %I:%M %p"),
            "Predicted_Probability": ten_year_score / 100,
            "Birthdate": birthdate.strftime("%Y-%m-%d"),
            "Age": age,
            "Gender": gender,
            "Race": race,
            "Total_Cholesterol": total_cholesterol,
            "HDL_Cholesterol": hdl_cholesterol,
            "Systolic_Blood_Pressure": systolic_bp,
            "Diabetes_Status": diabetes_status,
            "Current_Smoking_Status": current_smoking_status,
            "Hypertension_Treatment_Status": hypertension_treatment_status,
            "TenYearScore": ten_year_score,
            "LifetimeRisk": lifetime_risk,
            "LowestTenYear": lowest_ten_year,
            "LowestLifetime": lowest_lifetime,
            "PotentialRisk": potential_risk,
            "ASCVD_Actual_Outcome": actual_outcome,
            "Predicted_Outcome": ten_year_score / 100
        })

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("ASCVD_prediction_categorical_data.csv", index=False)

print("CSV file 'ASCVD_prediction_categorical_data.csv' created successfully!")
