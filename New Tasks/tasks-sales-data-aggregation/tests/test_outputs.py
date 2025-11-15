import os
import json
import pytest

# Define the path for the final report
REPORT_PATH = "/app/reports/product_summary.json"

def test_product_summary_file_exists():
    """
    Test 1: Check if the output JSON file was created at the correct path.
    """
    assert os.path.exists(REPORT_PATH), f"Expected file not found at {REPORT_PATH}"

def test_product_summary_content():
    """
    Test 2: Check if the content of the JSON report is correct.
    This test recalculates the expected output (the "golden" answer)
    and compares it to the agent's output.
    """
    # 1. Define the "golden" expected data.
    # Based on:
    # - Median price of (10.0, 10.0, 5.5, 25.0, 5.5) is 10.0
    # - PID-A45: (5*10.0) + (3*10.0) = 80.0 revenue, 5+3=8 units
    # - PID-B12: (2*10.0) + (4*25.0) = 120.0 revenue, 2+4=6 units
    # - PID-C78: (10*5.5) + (8*5.5) = 99.0 revenue, 10+8=18 units
    expected_data = [
        {"product_id": "PID-A45", "total_revenue": 80.0, "units_sold": 8},
        {"product_id": "PID-B12", "total_revenue": 120.0, "units_sold": 6},
        {"product_id": "PID-C78", "total_revenue": 99.0, "units_sold": 18}
    ]

    # 2. Load the agent's output file
    try:
        with open(REPORT_PATH, 'r') as f:
            agent_data = json.load(f)
    except json.JSONDecodeError:
        pytest.fail(f"Could not decode JSON from {REPORT_PATH}. File might be empty or malformed.")
    except IOError:
        pytest.fail(f"Could not open file {REPORT_PATH}. (Did it get created?)")

    # 3. Validate the content
    assert isinstance(agent_data, list), "Output JSON should be a list (records orientation)."
    assert len(agent_data) == len(expected_data), \
        f"Expected {len(expected_data)} product records, but found {len(agent_data)}."

    # Sort both lists by 'product_id' to ensure comparison is not order-dependent
    try:
        sorted_agent_data = sorted(agent_data, key=lambda x: x['product_id'])
        sorted_expected_data = sorted(expected_data, key=lambda x: x['product_id'])
    except KeyError:
        pytest.fail("The output JSON is missing the 'product_id' key in one or more records.")
    except TypeError:
         pytest.fail("Could not sort the output data. Ensure all records are dictionaries with a 'product_id'.")

    # 4. Compare the sorted lists
    for i, expected_row in enumerate(sorted_expected_data):
        agent_row = sorted_agent_data[i]
        
        # Check keys
        assert 'product_id' in agent_row, f"Row {i} missing 'product_id'"
        assert 'total_revenue' in agent_row, f"Row {i} missing 'total_revenue'"
        assert 'units_sold' in agent_row, f"Row {i} missing 'units_sold'"
        
        # Check values
        assert agent_row['product_id'] == expected_row['product_id'], \
            f"Product ID mismatch at row {i}"
        
        assert isinstance(agent_row['units_sold'], int), \
            f"Expected 'units_sold' for {agent_row['product_id']} to be an integer, got {type(agent_row['units_sold'])}"
        assert agent_row['units_sold'] == expected_row['units_sold'], \
            f"Value mismatch for 'units_sold' for {agent_row['product_id']}"
        
        # Use pytest.approx for floating point comparison
        assert agent_row['total_revenue'] == pytest.approx(expected_row['total_revenue']), \
            f"Value mismatch for 'total_revenue' for {agent_row['product_id']}"

    print(f"\nSuccessfully validated {REPORT_PATH}.")