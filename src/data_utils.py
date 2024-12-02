from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


def create_data_generator(dataframe, image_dir, target_size, batch_size, class_weights=None, shuffle=True):
    """
    Centralized function to create a data generator for training, validation, or testing.

    Args:
        dataframe (pd.DataFrame): DataFrame containing image paths and labels.
        image_dir (str): Path to the directory containing the images.
        target_size (tuple): Target size for resizing images (height, width).
        batch_size (int): Number of samples per batch.
        class_weights (dict, optional): Class weights for imbalanced data. Defaults to None.
        shuffle (bool, optional): Whether to shuffle the data per epoch. Defaults to True.

    Returns:
        generator: A data generator yielding (images, labels, sample_weights).
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
        shuffle=shuffle
    )

    if class_weights:
        # Define a generator with dynamic sample weights
        def generator_with_weights():
            for images, labels in base_generator:
                sample_weights = compute_sample_weights(labels, class_weights)
                yield images, labels.astype('int64'), sample_weights

        return generator_with_weights()

    # Default generator (without sample weights)
    return base_generator


def compute_sample_weights(labels, class_weights):
    """
    Computes sample weights based on class weights for each batch.

    Args:
        labels (np.ndarray): Labels for the batch (binary multi-label format).
        class_weights (dict): Class weights as a dictionary {class_index: weight}.

    Returns:
        np.ndarray: Array of sample weights for the batch.
    """
    sample_weights = np.zeros(labels.shape[0], dtype=np.float32)
    for i in range(labels.shape[1]):  # Iterate over each label (multi-label classification)
        sample_weights += labels[:, i] * class_weights[i]
    return sample_weights
