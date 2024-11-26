import pandas as pd
import numpy as np

def load_and_preprocess_labels(csv_path):
    """
    Loads and preprocesses the labels from the CSV file.
    - Excludes specific labels that are difficult to predict.
    - Ensures all label columns are numeric.
    - Handles mixed data types (e.g., 'YES', 'NO', 0, 1).
    - Cleans unexpected or null values.
    - Aggregates rows for the same Image ID by taking the union of labels.
    """
    # Load the CSV
    labels_df = pd.read_csv(csv_path)

    # Select relevant columns (excluding specific labels)
    labels_df = labels_df[['Image ID', 'Abnormal', 'Atelectasis', 'Cardiomegaly',
                           'Effusion', 'Infiltration', 'Pleural Thickening', 'Consolidation']]

    # Ensure all labels are cleaned and converted to binary
    def clean_label(value):
        """Converts 'YES', 1 to 1 and 'NO', 0 to 0; handles unexpected values."""
        if value in ['YES', 1]:
            return 1
        elif value in ['NO', 0]:
            return 0
        else:
            return 0  # Default to 0 for any unexpected or null value

    for col in labels_df.columns[1:]:
        labels_df[col] = labels_df[col].apply(clean_label)

    # Aggregate rows by Image ID (union of labels for each image)
    labels_df = labels_df.groupby('Image ID', as_index=False).max()

    # Ensure all label columns are of numeric type
    labels_df.iloc[:, 1:] = labels_df.iloc[:, 1:].astype(int)

    return labels_df
