import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate 1000 rows of synthetic data
num_rows = 1000

data = {
    "Age": np.random.randint(18, 91, num_rows),
    "Gender": np.random.choice(["Male", "Female"], num_rows),
    "Race": np.random.choice(["White", "Black", "Asian", "Hispanic", "Other"], num_rows),
    "Blood Sugar Level": np.random.randint(70, 201, num_rows),
    "Family Diabetes History": np.random.choice(["Yes", "No"], num_rows),
    "Training Label": np.random.choice([0, 1], num_rows, p=[0.7, 0.3]),  # 70% no diabetes, 30% diabetes
}

# Simulate predictions with some noise
data["Prediction Label"] = np.where(
    np.random.rand(num_rows) > 0.1,  # 90% of predictions match the training label
    data["Training Label"],
    1 - data["Training Label"]  # 10% flipped for error simulation
)

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("diabetes_prediction_data.csv", index=False)

print("CSV file 'diabetes_prediction_data.csv' created successfully!")
