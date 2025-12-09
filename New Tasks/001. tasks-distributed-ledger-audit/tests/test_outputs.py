import os
import pytest
import pandas as pd
import numpy as np

OUTPUT_PATH = "/app/audit/final_balances.parquet"
GOLDEN_PATH = "/app/task-deps/golden_balances.parquet"

def test_file_existence():
    """Confirms existence of the final required artifact."""
    assert os.path.exists(OUTPUT_PATH), f"File not found: {OUTPUT_PATH}"

def test_schema_enforcement():
    """
    Validates:
    1. Correct column names.
    2. Strict data types (Float64 for money, Int64 for counts).
    """
    try:
        df = pd.read_parquet(OUTPUT_PATH)
    except Exception as e:
        pytest.fail(f"Could not read parquet file: {str(e)}")

    expected_columns = {
        "account_id": ["object", "string"], # Allow pandas object or string dtype
        "final_balance": ["float64"],
        "transaction_count": ["int64"]
    }

    # Check Columns
    assert set(df.columns) == set(expected_columns.keys()), \
        f"Column mismatch. Expected {list(expected_columns.keys())}, got {list(df.columns)}"

    # Check Types
    for col, valid_types in expected_columns.items():
        # pandas dtype name check
        current_type = str(df[col].dtype)
        # Normalize some pandas type names
        if "string" in current_type: current_type = "string"
        if "object" in current_type: current_type = "object"
        
        assert current_type in valid_types, \
            f"Column '{col}' has invalid type '{df[col].dtype}'. Expected one of {valid_types}."

def test_row_order():
    """Validates the sort order requirement."""
    df = pd.read_parquet(OUTPUT_PATH)
    sorted_ids = df["account_id"].sort_values().reset_index(drop=True)
    current_ids = df["account_id"].reset_index(drop=True)
    pd.testing.assert_series_equal(current_ids, sorted_ids, obj="Sort Order")

def test_functional_correctness():
    """
    Compares the submission against the Golden Answer.
    This validates: Deduplication logic, LWW logic, Correction logic, and Tombstone logic.
    """
    student_df = pd.read_parquet(OUTPUT_PATH).set_index("account_id").sort_index()
    golden_df = pd.read_parquet(GOLDEN_PATH).set_index("account_id").sort_index()

    # Align indices (in case student missed accounts or added extra)
    assert len(student_df) == len(golden_df), \
        f"Row count mismatch. Expected {len(golden_df)} accounts, got {len(student_df)}."

    # Compare Balances (Tolerance 0.01 for float math)
    try:
        pd.testing.assert_series_equal(
            student_df["final_balance"], 
            golden_df["final_balance"], 
            atol=0.01, 
            obj="Final Balance"
        )
    except AssertionError as e:
        pytest.fail(f"Balance mismatch (Did you handle the CENTS in XML or CORRECTION events correctly?): {e}")

    # Compare Transaction Counts (Strict equality)
    # This specifically catches if Tombstones were merely zeroed out instead of removed.
    pd.testing.assert_series_equal(
        student_df["transaction_count"], 
        golden_df["transaction_count"], 
        obj="Transaction Count"
    )