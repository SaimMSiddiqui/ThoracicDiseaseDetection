import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from src.preprocess import load_and_preprocess_labels
from src.utils import verify_paths
from src.evaluate import evaluate_model

# Define the project root dynamically
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Paths
csv_file_path = os.path.join(
    PROJECT_ROOT,
    'gcs-public-data--healthcare-nih-chest-xray-labels',
    'all_findings_expert_labels',
    'all_findings_expert_labels_test_individual_readers.csv'
)
image_dir = os.path.join(PROJECT_ROOT, 'Labeled')
model_path = os.path.join(PROJECT_ROOT, 'Models', 'thoracic_disease_model.keras')

# Verify paths
verify_paths([csv_file_path, image_dir, model_path])

# Step 1: Load and preprocess data
print("\n### Loading and Preprocessing Labels ###")
labels_df = load_and_preprocess_labels(csv_file_path)
print(f"Data successfully loaded. Total records: {len(labels_df)}")

# Step 2: Split into train and validation sets
print("\n### Splitting Data ###")
_, val_df = train_test_split(labels_df, test_size=0.2, random_state=42)
print(f"Validation set size: {len(val_df)}")

# Step 3: Create validation data generator
print("\n### Creating Validation Generator ###")
batch_size = 32

def evaluation_data_generator(dataframe, image_dir, target_size, batch_size):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    return datagen.flow_from_dataframe(
        dataframe=dataframe,
        directory=image_dir,
        x_col='Image ID',
        y_col=dataframe.columns[1:].tolist(),
        target_size=target_size,
        batch_size=batch_size,
        class_mode='raw',
        shuffle=False  # Ensure reproducibility during evaluation
    )

val_gen = evaluation_data_generator(
    dataframe=val_df,
    image_dir=image_dir,
    target_size=(224, 224),
    batch_size=batch_size
)

# Calculate validation steps
val_steps = len(val_df) // batch_size
print(f"Validation steps: {val_steps}")

# Step 4: Load the saved model
print(f"\n### Loading Model from {model_path} ###")
model = load_model(model_path)
print("Model successfully loaded.")

# Step 5: Evaluate the model
evaluate_model(model, val_gen, steps=val_steps)
