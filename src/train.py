from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def compute_class_weights(dataframe):
    """
    Computes class weights to address class imbalance.

    Args:
        dataframe (pd.DataFrame): DataFrame containing binary labels.

    Returns:
        dict: Class weights as a dictionary {class_index: weight}.
    """
    labels = dataframe.iloc[:, 1:].values  # Exclude 'Image ID'
    num_labels = labels.shape[1]
    class_weights = {}

    for i in range(num_labels):
        label_column = labels[:, i]
        weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(label_column),
            y=label_column
        )
        class_weights[i] = (weights[0] + weights[1]) / 2  # Average weight for class 0 and 1
    return class_weights

def train_model(model, train_gen, val_gen, steps_per_epoch, validation_steps, epochs=20):
    """
    Trains the model using the provided generators.

    Args:
        model (keras.Model): Compiled model to train.
        train_gen (generator): Training data generator.
        val_gen (generator): Validation data generator.
        steps_per_epoch (int): Number of training steps per epoch.
        validation_steps (int): Number of validation steps per epoch.
        epochs (int): Total number of epochs.

    Returns:
        History: Training history object.
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print("\n### Starting Model Training ###")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=[early_stopping]
    )
    return history
