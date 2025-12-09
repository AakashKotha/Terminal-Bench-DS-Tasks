import os
import joblib
import numpy as np
import pytest
from sklearn.base import is_classifier
from sklearn.linear_model import LogisticRegression

# Define paths
MODEL_PATH = "/app/models/model.joblib"
PREPROCESS_SCRIPT_PATH = "/app/src/preprocess.py"
TRAIN_SCRIPT_PATH = "/app/src/train.py"
PROCESSED_DATA_PATH = "/app/data/processed_data.npz"
HIDDEN_TEST_SET_PATH = "/app/tests/hidden_test_set.npz"
MIN_ACCURACY_THRESHOLD = 0.90

@pytest.fixture(scope="module")
def agent_model():
    """Fixture to load the agent's saved model once."""
    if not os.path.exists(MODEL_PATH):
        pytest.fail(f"Model file not found at {MODEL_PATH}. "
                    "Did the modified train.py script run successfully?")
    
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        pytest.fail(f"Failed to load model from {MODEL_PATH}. Error: {e}")

def test_required_files_exist():
    """Test 1: Check that all required output files were created."""
    assert os.path.exists(PREPROCESS_SCRIPT_PATH), \
        "The preprocessing script `/app/src/preprocess.py` was not created."
    
    assert os.path.exists(PROCESSED_DATA_PATH), \
        "The processed data file `/app/data/processed_data.npz` was not created."
        
    assert os.path.exists(MODEL_PATH), \
        "The final model file `/app/models/model.joblib` was not created."

def test_model_type(agent_model):
    """Test 2: Check if the saved model is a scikit-learn classifier."""
    assert is_classifier(agent_model), "The saved file is not a valid scikit-learn classifier."
    # We can be more specific if we want
    assert isinstance(agent_model, LogisticRegression), \
        "The saved model is not a LogisticRegression instance."

def test_model_performance_on_hidden_set(agent_model):
    """
    Test 3: The critical test. Validate the model's performance on a
    hidden, correctly processed test set.
    """
    
    # 1. Load the hidden test set
    # This file was created by the builder stage and is not visible to the agent's
    # scripts (it's not in /app/data).
    try:
        data = np.load(HIDDEN_TEST_SET_PATH)
        X_test_hidden = data['X_test']
        y_test_hidden = data['y_test']
    except Exception as e:
        pytest.fail(f"Failed to load the hidden test set. Test harness error: {e}")

    print(f"\nLoaded hidden test set with {X_test_hidden.shape[0]} samples.")
    
    # 2. Get the number of features the agent's model *expects*
    # A correctly preprocessed model will expect 5 features:
    # 2 (scaled numerical) + 3 (one-hot categorical)
    try:
        n_features_in = agent_model.n_features_in_
    except AttributeError:
        pytest.fail("Could not determine `n_features_in_` from the loaded model. "
                    "Was the model trained?")

    # 3. Check feature mismatch
    # This is a key validation step.
    n_features_expected = X_test_hidden.shape[1]
    assert n_features_in == n_features_expected, \
        f"Model feature mismatch. The model was trained on {n_features_in} features, " \
        f"but the correctly processed data has {n_features_expected} features. " \
        "Did you correctly one-hot-encode 'department' and use all features?"

    # 4. Score the model
    try:
        accuracy = agent_model.score(X_test_hidden, y_test_hidden)
    except Exception as e:
        pytest.fail(f"Model failed to predict on the test set. Error: {e}")

    print(f"Model accuracy on hidden test set: {accuracy:.4f}")
    
    # 5. Assert performance
    assert accuracy >= MIN_ACCURACY_THRESHOLD, \
        f"Model accuracy ({accuracy:.4f}) is below the required threshold " \
        f"({MIN_ACCURACY_THRESHOLD}). This indicates the data was not " \
        "preprocessed correctly (e.g., features not scaled or encoded)."

    print("\n--- Model Performance Test Passed ---")