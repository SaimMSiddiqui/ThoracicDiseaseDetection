from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def create_data_generator(dataframe, image_dir, target_size, batch_size, class_weights):
    """
    Creates a data generator for training/validation data and dynamically applies class weights to sample weights.
    """
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    base_generator = datagen.flow_from_dataframe(
        dataframe=dataframe,
        directory=image_dir,
        x_col='Image ID',
        y_col=dataframe.columns[1:].tolist(),
        target_size=target_size,
        batch_size=batch_size,
        class_mode='raw',
        shuffle=True  # Shuffle the data per epoch
    )

    # Add dynamic sample weights to the generator output
    def generator_with_weights():
        for images, labels in base_generator:
            # Calculate sample weights based on class weights
            sample_weights = np.zeros(labels.shape[0], dtype=np.float32)  # Initialize sample weights
            for i in range(labels.shape[1]):  # Iterate over each label
                sample_weights += labels[:, i] * class_weights[i]  # Add class weights for active labels
            yield images, labels.astype('int64'), sample_weights

    return generator_with_weights()



def compute_class_weights(dataframe):
    """
    Computes class weights to address class imbalance.
    """
    # Flatten the labels to calculate class frequencies
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
        # Average weights for class 0 and 1
        class_weights[i] = (weights[0] + weights[1]) / 2  # One weight per label

    return class_weights


def train_model(model, train_gen, val_gen, steps_per_epoch, validation_steps, epochs=20):
    """
    Trains the model using the provided generators.
    """
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
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



