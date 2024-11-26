from sklearn.metrics import classification_report, hamming_loss
import numpy as np


def evaluate_model(model, test_gen, steps):
    """
    Evaluates the model on test data and computes additional metrics.
    Args:
        model: Trained model to evaluate (either freshly trained or loaded from file).
        test_gen: Test data generator (may or may not include sample weights).
        steps: Number of steps to evaluate (batches in the test set).
    """
    # Evaluate model using the generator
    print("\n### Evaluating Model ###")
    test_loss, test_acc = model.evaluate(test_gen, steps=steps)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")

    # Generate predictions
    y_pred = []
    y_true = []

    print("\n### Generating Predictions ###")
    for i, batch in enumerate(test_gen):
        if len(batch) == 3:  # Check if sample weights are included
            images, labels, _ = batch  # Unpack with weights
        else:
            images, labels = batch  # Unpack without weights

        y_pred.append(model.predict(images))  # Predict probabilities
        y_true.append(labels)

        if i + 1 >= steps:  # Break when all steps are processed
            break

    # Convert to arrays
    y_pred = np.vstack(y_pred)
    y_true = np.vstack(y_true)

    # Binarize predictions (threshold > 0.5 for multi-label classification)
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Compute additional metrics
    print("\n### Classification Report ###")
    print(classification_report(y_true, y_pred_binary))
    print(f"\nHamming Loss: {hamming_loss(y_true, y_pred_binary)}")
