import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

print("Generating data for the task...")

# 1. Create a synthetic dataset that *requires* scaling and encoding
X, y = make_classification(
    n_samples=200,
    n_features=2,        # We will add a 3rd categorical one
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    flip_y=0.05,
    class_sep=1.5,
    random_state=42
)

# 2. "De-scale" the features to create the "trap"
# Feature 0 -> salary (large scale)
# Feature 1 -> projects_completed (small scale)
X[:, 0] = (X[:, 0] + 1.5) * 25000 + 50000  # Scale to a salary range
X[:, 1] = (X[:, 1] + 3) * 1.5 + 1          # Scale to a project count range

# 3. Add a categorical feature that is informative
departments = []
for i in range(200):
    if (X[i, 1] + y[i]) % 3 == 0:
        departments.append("Sales")
    elif (X[i, 1] + y[i]) % 3 == 1:
        departments.append("IT")
    else:
        departments.append("HR")

# 4. Create the final "dirty" DataFrame
df = pd.DataFrame(X, columns=['salary', 'projects_completed'])
df['department'] = departments
df['will_be_promoted'] = y
df['employee_id'] = range(1001, 1201)

# Reorder columns
df = df[['employee_id', 'department', 'salary', 'projects_completed', 'will_be_promoted']]

# 5. Split into the agent's data and the hidden test set
train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)

# Save the agent's "dirty" data
train_df.to_csv("employees.csv", index=False)
print("Saved employees.csv (for agent)")

# 6. Create the "golden" hidden test set
# This is what the test script will use to validate the agent's model
print("Generating hidden test set...")

# Define the *correct* preprocessing pipeline
categorical_features = ['department']
numerical_features = ['salary', 'projects_completed']

# Create the transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ],
    remainder='drop'
)

# NOTE: We FIT the preprocessor on the TRAINING data
# This is a critical step.
X_train_dirty = train_df.drop('will_be_promoted', axis=1)
y_train_dirty = train_df['will_be_promoted']
preprocessor.fit(X_train_dirty)

# Now, we TRANSFORM the HIDDEN TEST data
X_test_hidden_dirty = test_df.drop('will_be_promoted', axis=1)
y_test_hidden = test_df['will_be_promoted']

X_test_processed = preprocessor.transform(X_test_hidden_dirty)
y_test_processed = y_test_hidden.values

# Save the *processed* hidden test set
np.savez_compressed(
    "hidden_test_set.npz",
    X_test=X_test_processed,
    y_test=y_test_processed
)
print("Saved hidden_test_set.npz (for test harness)")

print("Data generation complete.")