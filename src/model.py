from keras.src.utils.module_utils import tensorflow
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def build_model(input_shape, num_labels, trainable_layers=10, dense_units=128, dropout_rate=0.5):
    """
    Builds and compiles the ResNet-based model with improvements:
    Uses a ResNet Base model and a custom output layer to have image recognition functionality
    plus the ability to train to the specified dataset.

    - Supports fine-tuning a specified number of layers in ResNet50.
    - Adds L2 regularization and dropout for improved generalization.
    - Reduces the complexity of the dense layers.

    Args:
        input_shape (tuple): Shape of the input images (e.g., (224, 224, 3)).
        num_labels (int): Number of output labels (multi-label classification).
        trainable_layers (int): Number of trainable layers in the base model.
        dense_units (int): Number of units in the dense layer.
        dropout_rate (float): Dropout rate for regularization.

    Returns:
        model (tensorflow.keras.Model): Compiled Keras model.
    """
    # Load the ResNet50 base model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze all layers initially
    for layer in base_model.layers:
        layer.trainable = False

    # Unfreeze the last 'trainable_layers' layers for fine-tuning
    if trainable_layers > 0:
        for layer in base_model.layers[-trainable_layers:]:
            layer.trainable = True

    # Create custom layers______________________________________# Details
    base_output = base_model.output                             # Output from the ResNet50 base model
    pooled_features = GlobalAveragePooling2D()(base_output)     # Features after global average pooling
    dense_output = Dense(dense_units, activation='relu',        # Dense layer with L2 regularization
        kernel_regularizer=l2(0.01))(pooled_features)           # --
    regularized_output = Dropout(dropout_rate)(dense_output)    # Regularized output with dropout
    final_output = Dense(num_labels, activation='sigmoid')(     # --
        regularized_output)                                     # Final output layer with sigmoid activation

    # Combine base layer with custom layer
    model = Model(inputs=base_model.input, outputs=final_output)

    # Dynamic learning rate
    lr_schedule = tensorflow.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=10000,
        decay_rate=0.9
    )
    optimizer = Adam(learning_rate=lr_schedule)

    # Compile the model with Adam optimizer
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
