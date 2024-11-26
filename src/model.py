from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def build_model(input_shape, num_labels, trainable_layers=10, dense_units=128, dropout_rate=0.5):
    """
    Builds and compiles the ResNet-based model with improvements:
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

    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Global average pooling
    x = Dense(dense_units, activation='relu', kernel_regularizer=l2(0.01))(x)  # Dense layer with L2 regularization
    x = Dropout(dropout_rate)(x)  # Dropout for regularization
    output = Dense(num_labels, activation='sigmoid')(x)  # Sigmoid for multi-label classification

    # Define the model
    model = Model(inputs=base_model.input, outputs=output)

    # Compile the model with Adam optimizer
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
