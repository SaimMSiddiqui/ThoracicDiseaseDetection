import math
import os
from src.data_utils import create_data_generator
from src.preprocess import load_and_preprocess_labels
from src.model import build_model
from src.train import train_model, compute_class_weights
from src.evaluate import evaluate_model
from src.utils import verify_paths
from sklearn.model_selection import train_test_split

# Define the project root dynamically
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to the project root
csv_file_path = os.path.join(
    PROJECT_ROOT,
    'gcs-public-data--healthcare-nih-chest-xray-labels',
    'all_findings_expert_labels',
    'all_findings_expert_labels_test_individual_readers.csv'
)

image_dir = os.path.join(PROJECT_ROOT, 'Labeled')

# Verify paths
verify_paths([csv_file_path, image_dir])

# Load and preprocess data
labels_df = load_and_preprocess_labels(csv_file_path)

# Analyze label distribution
print("\nLabel Distribution:")
label_counts = labels_df.iloc[:, 1:].sum(axis=0)  # Sum of each label across rows
print(label_counts)
print("\nNumber of images with multiple labels:")
multi_label_count = (labels_df.iloc[:, 1:].sum(axis=1) > 1).sum()
print(multi_label_count)

# Compute class weights
class_weights = compute_class_weights(labels_df)
print(f"\nClass Weights: {class_weights}")

# Train-validation split
train_df, val_df = train_test_split(labels_df, test_size=0.2, random_state=42)

# Calculate steps per epoch dynamically
train_steps_per_epoch = math.ceil(len(train_df) / 32)  # Batch size = 32
val_steps_per_epoch = math.ceil(len(val_df) / 32)  # Batch size = 32

# Print debug info to confirm correct steps
print(f"Train steps per epoch: {train_steps_per_epoch}")
print(f"Validation steps per epoch: {val_steps_per_epoch}")

# Create data generators for training and evaluation
train_gen = create_data_generator(
    dataframe=train_df,
    image_dir=image_dir,
    target_size=(224, 224),
    batch_size=32,
    class_weights=class_weights,
    shuffle=True  # Shuffle training data
)

val_gen = create_data_generator(
    dataframe=val_df,
    image_dir=image_dir,
    target_size=(224, 224),
    batch_size=32,
    class_weights=None,  # No class weights for evaluation
    shuffle=False  # Do not shuffle validation data
)

# Dynamically calculate the number of labels
num_labels = len(labels_df.columns) - 1  # Exclude 'Image ID'

# Build the model with dynamically determined output size
model = build_model(
    input_shape=(224, 224, 3),
    num_labels=num_labels,  # Use dynamic label count
    trainable_layers=10,
    dense_units=128,
    dropout_rate=0.5
)

# Train the model
print("\n### Starting Model Training ###")
history = train_model(
    model=model,
    train_gen=train_gen,
    val_gen=val_gen,
    steps_per_epoch=train_steps_per_epoch,
    validation_steps=val_steps_per_epoch,
    epochs=20
)

# Save the trained model
save_dir = 'Models/'
os.makedirs(save_dir, exist_ok=True)
model.save(os.path.join(save_dir, 'thoracic_disease_model.keras'))
print("Model saved in TensorFlow SavedModel format.")

# Get the label names (excluding 'Image ID')
label_names = labels_df.columns[1:].tolist()

# Evaluate the model
print("\n### Evaluating Model on Validation Set ###")
evaluate_model(model, val_gen, steps=val_steps_per_epoch, label_names=label_names)
