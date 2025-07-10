# -*- coding: utf-8 -*-

"""
======================================================================
Script Name: test_bonn_paradigm.py
Description: An independent test script for processing and exporting the Bonn EEG dataset (Pytest compatible).
======================================================================

This script is designed to independently test the full functionality of the
`export_eeg_data` function and is structured to be automatically
discovered and executed by the Pytest testing framework.

It performs the following operations:
1.  Loads data from the specified Bonn EEG dataset path.
2.  Processes the data using a paradigm defined in the `metabci` library.
3.  Performs shape validation and numerical scaling on the data.
4.  Sorts the data according to event labels and subject IDs.
5.  Exports the processed and sorted signal data and label data into separate JSON files.
6.  Verifies the success of the entire process using an `assert` statement.

---
**How to Run:**
1.  **Via Pytest (Recommended):**
    Run the `pytest` command in the terminal from the project's root directory.
    Pytest will automatically find and execute the `test_full_data_export_process`
    function below.

2.  **As a Standalone Script:**
    Run `python test_bonn_paradigm.py` in the terminal from the project's root
    directory. The script will execute the test via the `if __name__ == '__main__':` block.

---
**Prerequisites:**
1.  Ensure all required libraries (numpy, pandas, metabci) are correctly
    installed in your Python environment.
2.  Ensure the Bonn EEG dataset has been downloaded and is located at
    `Dataset/Bonn_EEG/` relative to the script's location.
3.  This script should be located in your project's root directory.
---
"""

import numpy as np
import pandas as pd
import json
import traceback
import os
import sys

# Assume the `metabci` library is installed and in Python's search path
try:
    from metabci.brainda.datasets.bonn_eeg import BonnEEGDataset
    from metabci.brainda.paradigms.Bonn_paradigm import BonnEEGParadigm
except ImportError as e:
    print("=" * 60)
    print("🔥 Critical Error: Could not import the 'metabci' library.")
    print("Please ensure you have installed the project dependencies correctly and are running this script from the project root.")
    print(f"Detailed error message: {e}")
    print("=" * 60)
    sys.exit(1)  # Critical error, exit immediately


def export_eeg_data():
    """
    Core function to load, process, sort, and export the Bonn EEG dataset.
    Returns True if all steps succeed, otherwise returns False.
    """
    try:
        print("=" * 60)
        print("🚀 Starting the core data export task...")
        print("=" * 60)

        # --- Step 1: Check path and load data ---
        print("--- Step 1: Checking path and loading data... ---")
        dataset_path = 'Dataset/Bonn_EEG/'
        if not os.path.isdir(dataset_path):
            print(f"❌ Error: Dataset directory not found at '{dataset_path}'.")
            print("Please ensure the Bonn EEG dataset is located at the correct path.")
            return False

        print("This may take some time as we are processing all 300 files.")
        all_subjects = list(range(1, 101))
        events_with_labels = {'O': 0, 'F': 1, 'S': 2}
        dataset = BonnEEGDataset(path=dataset_path)
        paradigm = BonnEEGParadigm(events=events_with_labels)

        x, y, meta = paradigm.get_data(
            dataset=dataset,
            subjects=all_subjects,
            return_concat=True,
            n_jobs=-1,
            verbose=False,
        )
        print("✅ Data loaded successfully!")

        # --- Step 2: Validate and scale data ---
        print("\n--- Step 2: Validating and scaling data ---")
        expected_trials = len(all_subjects) * len(events_with_labels)
        assert x.shape == (expected_trials, 1, 4096), f"Data x has an unexpected shape! Expected ({expected_trials}, 1, 4096), but got {x.shape}"
        assert y.shape == (expected_trials,), f"Labels y have an unexpected shape! Expected ({expected_trials},), but got {y.shape}"
        print("✅ Data shape validation passed.")

        x = x / 1e6
        print("✅ Data scaled back to original value range by dividing by 1e6.")

        # --- Step 3: Sort data by label groups ---
        print("\n--- Step 3: Sorting data by event (label) and subject ---")
        meta.sort_values(by=['event', 'subject'], inplace=True)
        sorted_indices = meta.index
        x_sorted = x[sorted_indices]
        y_sorted = y[sorted_indices]
        print("✅ Data has been reordered by event type and subject ID.")

        # --- Step 4: Export to JSON files ---
        print("\n--- Step 4: Exporting to JSON files at the specified path ---")
        output_dir = os.path.join('metabci', 'brainda', 'data')
        os.makedirs(output_dir, exist_ok=True)
        print(f"✅ Ensured output directory exists: '{output_dir}'")

        sig_data_filename = os.path.join(output_dir, 'sigData.json')
        label_data_filename = os.path.join(output_dir, 'labelData.json')

        print(f"⏳ Saving signal data to '{sig_data_filename}'...")
        with open(sig_data_filename, 'w') as f:
            json.dump(x_sorted.tolist(), f)
        print(f"✅ '{sig_data_filename}' saved successfully!")

        print(f"⏳ Saving label data to '{label_data_filename}'...")
        with open(label_data_filename, 'w') as f:
            json.dump(y_sorted.tolist(), f)
        print(f"✅ '{label_data_filename}' saved successfully!")

        print("\n🎉 Core data export task completed successfully!")
        return True

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"🔥 Critical Error: An exception occurred during data export, task aborted.")
        print("=" * 60)
        traceback.print_exc()
        return False


# =======================================================================
# Pytest Test Case
# Pytest will automatically discover and run this function starting with 'test_'.
# =======================================================================
def test_full_data_export_process():
    """
    This is a Pytest-discoverable test case.
    It encapsulates the entire data export process and uses an assertion
    to verify its final result.
    """
    print("\n" + "=" * 70)
    print("###      Executing Bonn EEG Data Export Pytest Case      ###")
    print("=" * 70)

    # Call the core functional-part
    is_success = export_eeg_data()

    # Use an assert statement to verify the result
    # This is the key to the test: if is_success is False, the test will fail and report an error.
    assert is_success is True, "The data export process (export_eeg_data) returned False, indicating an error occurred. Please check the log output above."

    print("\n" + "=" * 70)
    print("✅✅✅ Pytest assertion passed: export_eeg_data() returned True successfully.")
    print("Please check for 'sigData.json' and 'labelData.json' in the 'metabci/brainda/data/' directory.")
    print("=" * 70)


# =======================================================================
# Main Execution Block
# This allows the script to be run both by `pytest` and as a normal Python script.
# =======================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("###      Running data export test in standalone script mode      ###")
    print("=" * 70)

    # Call the test case function directly
    test_full_data_export_process()

    # Pause at the end to allow the user to see the full log output in the command line
    input("\nTest process finished. Press Enter to exit...")