import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# Define file paths
DATA_PATH = "/app/data/employees.csv"
MODEL_DIR = "/app/models"
MODEL_PATH = "/app/models/model.joblib"

# Ensure the model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Starting model training using data from {DATA_PATH}...")

# Load the "dirty" data
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}")
    exit(1)

# ---
# THIS IS THE "BROKEN" PART OF THE SCRIPT
# It ignores the 'department' column because it's a string.
# It does not scale the numerical features.
# ---
print("Using unscaled numerical features: 'salary', 'projects_completed'")
features = ['salary', 'projects_completed']
target = 'will_be_promoted'

try:
    X = df[features]
    y = df[target]
except KeyError:
    print("Error: Required columns not found in the CSV.")
    exit(1)


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
print("Training LogisticRegression model...")
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print(f"\n--- Initial Model Evaluation (on unscaled data) ---")
print(f"Accuracy: {acc:.4f}")
print("--------------------------------------------------")

# Save the model
print(f"Saving model to {MODEL_PATH}...")
joblib.dump(model, MODEL_PATH)

print("Training script finished.")