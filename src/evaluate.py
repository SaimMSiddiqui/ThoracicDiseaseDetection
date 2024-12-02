from sklearn.metrics import classification_report, hamming_loss
import numpy as np

def evaluate_model(model, test_gen, steps, label_names):
    """
    Evaluates the model on test data and computes additional metrics.

    Args:
        model: Trained model to evaluate.
        test_gen: Test data generator.
        steps: Number of steps to evaluate (batches in the test set).
        label_names: List of label names for classification report.
    """
    print("\n### Evaluating Model ###")
    test_loss, test_acc = model.evaluate(test_gen, steps=steps)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")

    y_pred, y_true = [], []

    print("\n### Generating Predictions ###")
    for i, (images, labels) in enumerate(test_gen):
        y_pred.append(model.predict(images))
        y_true.append(labels)
        if i + 1 >= steps:
            break

    y_pred = np.vstack(y_pred)
    y_true = np.vstack(y_true)

    y_pred_binary = (y_pred > 0.5).astype(int)

    print("\n### Classification Report ###")
    print(classification_report(y_true, y_pred_binary, target_names=label_names))
    print(f"\nHamming Loss: {hamming_loss(y_true, y_pred_binary)}")
