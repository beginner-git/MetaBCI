import numpy as np
import pandas as pd
import json
import traceback
import os
import sys
import subprocess


def export_eeg_data():
    """
    Loads, processes, sorts, and exports the Bonn EEG dataset.
    Returns True if all steps are successful, otherwise returns False.
    """
    try:
        print("=" * 60)
        print("🚀 Part 1: Starting data export task...")
        print("=" * 60)
        print("--- Step 1: Loading data... ---")
        print("This may take a while as we are processing all 300 files.")

        from metabci.brainda.datasets.bonn_eeg import BonnEEGDataset
        from metabci.brainda.paradigms.Bonn_paradigm import BonnEEGParadigm

        all_subjects = list(range(1, 101))
        events_with_labels = {'O': 0, 'F': 1, 'S': 2}
        dataset = BonnEEGDataset(path='Dataset/Bonn_EEG/')
        paradigm = BonnEEGParadigm(events=events_with_labels)

        x, y, meta = paradigm.get_data(
            dataset=dataset,
            subjects=all_subjects,
            return_concat=True,
            n_jobs=-1,
            verbose=False,
        )
        print("✅ Data loaded successfully!")

        # --- Step 2: Data validation and scaling ---
        print("\n--- Step 2: Validating and scaling data ---")
        expected_trials = len(all_subjects) * len(events_with_labels)
        assert x.shape == (expected_trials, 1, 4096), "The shape of data x is not as expected!"
        assert y.shape == (expected_trials,), "The shape of label y is not as expected!"
        print("✅ Data shape validation passed.")

        x = x / 1e6
        print("✅ Data has been scaled back to its original value range by dividing by 1e6.")

        # --- Step 3: Sort by label group ---
        print("\n--- Step 3: Sorting data by event (label) and subject ---")
        meta.sort_values(by=['event', 'subject'], inplace=True)
        sorted_indices = meta.index
        x_sorted = x[sorted_indices]
        y_sorted = y[sorted_indices]
        print("✅ Data has been reordered according to event type and subject ID.")

        # --- Step 4: Export to JSON files at the specified path ---
        print("\n--- Step 4: Exporting to JSON files at the specified path ---")
        output_dir = os.path.join('metabci', 'brainda', 'data')
        os.makedirs(output_dir, exist_ok=True)
        print(f"✅ Ensuring output directory exists: '{output_dir}'")

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

        print("\n🎉 Data export task completed successfully!")
        return True

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"🔥 ERROR: A critical error occurred during data export. Cannot continue.")
        print("=" * 60)
        traceback.print_exc()
        return False



def run_gui_application():
    """
    A robust launcher to run the GUI application in the subproject
    from the main project directory.
    """
    try:
        print("\n" + "=" * 60)
        print("🚀 Part 2: Preparing to launch the GUI application...")
        print("=" * 60)

        # Get the directory where this launcher script is located (i.e., the project root)
        launcher_dir = os.path.dirname(os.path.abspath(__file__))

        # Build the path to the GUI subproject directory and its main script
        gui_project_dir = os.path.join(launcher_dir, 'metabci', 'brainda', 'GUI')
        app_script_path = os.path.join(gui_project_dir, 'main.py')

        # --- Provide user-friendly error checking ---
        if not os.path.isdir(gui_project_dir):
            print(f"❌ ERROR: GUI subproject directory not found.")
            print(f"   Expected path: {gui_project_dir}")
            return
        if not os.path.isfile(app_script_path):
            print(f"❌ ERROR: Main script 'main.py' not found in the GUI directory.")
            print(f"   Expected path: {app_script_path}")
            return

        # --- Execute launch ---
        print(f"✅ GUI application found, preparing to launch...")
        print(f"   > Target directory: {gui_project_dir}")
        print(f"   > Execution script: {app_script_path}")

        # Use subprocess.run to execute the script
        # 'cwd' parameter is key here, setting the working directory of the child process to the GUI's directory
        # 'sys.executable' ensures we use the same Python interpreter as the launcher
        subprocess.run(
            [sys.executable, app_script_path],
            cwd=gui_project_dir,
            check=True  # This will raise an exception if the child process returns a non-zero exit code
        )

    except subprocess.CalledProcessError as e:
        print("\n" + "-" * 60)
        print(f"❗️ GUI application exited unexpectedly with return code: {e.returncode}")
    except Exception as e:
        print("\n" + "-" * 60)
        print(f"🔥 An unknown error occurred in the GUI launcher: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    # First, execute the data export function
    is_data_ready = export_eeg_data()

    # Then, check if the data is ready. If so, launch the GUI
    if is_data_ready:
        run_gui_application()
    else:
        print("\n" + "=" * 60)
        print("❌ Task aborted: The GUI application will not be launched due to data export failure.")
        print("=" * 60)
        # Pause to allow the user to see the error message
        input("Press Enter to exit...")


