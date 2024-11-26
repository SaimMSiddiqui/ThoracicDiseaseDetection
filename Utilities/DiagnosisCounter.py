import pandas as pd
import os

# Relative path to the CSV file from the content root
csv_file = 'gcs-public-data--healthcare-nih-chest-xray-labels/all_findings_expert_labels/all_findings_expert_labels_test_individual_readers.csv'

# Check if the file exists
if not os.path.exists(csv_file):
    print("File not found. Please check the path relative to the project root.")
else:
    print("File found. Proceeding to load...")

    # Load the CSV file
    df = pd.read_csv(csv_file)

    # List of diagnoses
    diagnoses = df.columns[4:]  # Assuming diagnoses start from the 5th column onward

    # Count occurrences of each diagnosis
    diagnosis_counts = {}
    for diagnosis in diagnoses:
        # Count rows where the diagnosis is 'YES'
        diagnosis_counts[diagnosis] = (df[diagnosis] == 'YES').sum()

    # Print the results
    print("\nDiagnosis counts:")
    for diagnosis, count in diagnosis_counts.items():
        print(f"{diagnosis}: {count}")
