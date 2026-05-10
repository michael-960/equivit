import numpy as np
from typing import Literal


class ClassificationMetric:
    def __call__(self, confmat: np.ndarray) -> float:
        """Computes the metric based on the provided confusion matrix.
        
        Args:
            confmat: A 2D numpy array representing the confusion matrix.
        
        Returns:
            The computed metric as a float.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class Accuracy(ClassificationMetric):
    def __call__(self, confmat: np.ndarray) -> float:
        """Computes accuracy from the confusion matrix."""
        if confmat.size == 0:
            return 0.0
        correct = np.trace(confmat)
        total = confmat.sum()
        return correct / total if total > 0 else 0.0


class Precision(ClassificationMetric):
    def __call__(self, confmat: np.ndarray) -> float:
        """Computes precision from the confusion matrix."""
        if confmat.size == 0:
            return 0.0
        true_positives = np.diag(confmat)
        predicted_positives = confmat.sum(axis=0)
        precision_per_class = np.divide(true_positives, predicted_positives, out=np.zeros_like(true_positives, dtype=float), where=predicted_positives!=0)

        return np.mean(precision_per_class)


class Recall(ClassificationMetric):
    def __call__(self, confmat: np.ndarray) -> float:
        """Computes recall from the confusion matrix."""
        if confmat.size == 0:
            return 0.0
        true_positives = np.diag(confmat)
        actual_positives = confmat.sum(axis=1)
        recall_per_class = np.divide(true_positives, actual_positives, out=np.zeros_like(true_positives, dtype=float), where=actual_positives!=0)

        return np.mean(recall_per_class)