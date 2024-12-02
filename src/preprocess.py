import pandas as pd

def load_and_preprocess_labels(csv_path):
    """
    This function prepares image labels for machine learning by cleaning, standardizing,
    and aggregating data from a CSV file. It ensures the labels are in a consistent format,
    reducing potential sources of error in training. The resulting dataset is well-suited
    for multi-label classification tasks, a common approach in medical imaging applications.

    Loads and preprocesses the labels from the CSV file.
    - Excludes specific labels that are difficult to predict.
    - Ensures all label columns are numeric.
    - Handles mixed data types (e.g., 'YES', 'NO', 0, 1).
    - Cleans unexpected or null values.
    - Aggregates rows for the same Image ID by taking the union of labels.
    """
    # Load the CSV
    labels_df = pd.read_csv(csv_path)

    # Selected columns to train
    labels_df = labels_df[['Image ID', 'Abnormal', 'Atelectasis', 'Nodule',
                           'Effusion', 'Pleural Thickening', 'Consolidation',
                           'Mass', 'Pneumothorax']]

    # Ensure all labels are cleaned and converted to binary
    def clean_label(value):
        #Converts 'YES' to 1 and 'NO' to 0; handles unexpected values.
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
