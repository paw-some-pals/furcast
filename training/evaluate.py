"""
Model evaluation utilities works with any fitted sklearn pipeline.
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    mean_absolute_error, root_mean_squared_error,
)


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate a fitted pipeline on a held-out test set.

    Returns a metrics dict. For numeric targets: mae, rmse.
    For categorical targets: accuracy, report.
    """
    # Predict on the test set
    y_pred = model.predict(X_test)
    # Determine if it's a regression or classification problem based on the target dtype
    is_regression = pd.api.types.is_numeric_dtype(y_test)

    if is_regression:
        metrics = {
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": root_mean_squared_error(y_test, y_pred),
        }
        print(f"MAE:  {metrics['mae']:.2f}")
        print(f"RMSE: {metrics['rmse']:.2f}")
    else:
        cm = confusion_matrix(y_test, y_pred)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "report": classification_report(y_test, y_pred),
            "confusion_matrix": cm,
        }
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(metrics["report"])
        disp = ConfusionMatrixDisplay(cm, display_labels=sorted(y_test.unique()))
        disp.plot(xticks_rotation="vertical")
        plt.tight_layout()
        plt.show()

    return metrics
