import numpy as np
import torch


# maybe we will move to torchmetrics in the future

class ConfusionMatrixCalculator:
    """
    A class to calculate and accumulate a confusion matrix for classification tasks.

    This class maintains an internal confusion matrix that can dynamically
    resize to accommodate new classes as they are encountered in the data. It
    provides methods to update the confusion matrix with new batches of ground truth and
    predicted labels, reset the matrix, and retrieve the accumulated confusion
    matrix.
    """
    def __init__(self):
        """Initializes the calculator with an empty confusion matrix."""
        self.reset_confusion_matrix()

    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
        """
        Calculates the confusion matrix for the new data and accumulates it.
        Dynamically resizes the internal matrix if new classes are encountered.
        
        Args:
            y_true: Ground truthlabels for the incoming batch.
            y_pred: Predicted labels for the incoming batch.
        """
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true has shape {y_true.shape} but y_pred has shape {y_pred.shape}")

        if y_pred.device != self.confusion_matrix.device:
            # this should only happen on the first update, but we handle it just in case
            self.confusion_matrix = self.confusion_matrix.to(y_pred.device)

        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
        
        # Guard against empty batches
        if len(y_true) == 0 or len(y_pred) == 0:
            return

        # Find the highest class index in the incoming data
        batch_num_classes = max(y_true.max().item(), y_pred.max().item()) + 1
        
        current_num_classes = self.confusion_matrix.shape[0]
        new_num_classes = max(current_num_classes, batch_num_classes)

        # Dynamically resize the stored matrix if the new data has more classes
        if new_num_classes > current_num_classes:
            padded_cm = self.confusion_matrix.new_zeros((new_num_classes, new_num_classes), dtype=torch.int64)
            # Copy old data into the top-left corner
            if current_num_classes > 0:
                padded_cm[:current_num_classes, :current_num_classes] = self.confusion_matrix
            self.confusion_matrix = padded_cm

        # Compute the confusion matrix for the new batch
        # We use `new_num_classes` so its shape matches the updated self.confusion_matrix
        linear_indices = y_true * new_num_classes + y_pred
        batch_cm_1d = torch.bincount(linear_indices, minlength=new_num_classes**2)
        batch_cm = batch_cm_1d.reshape(new_num_classes, new_num_classes)

        # Add the new batch's results to the accumulated matrix
        self.confusion_matrix += batch_cm

    def reset_confusion_matrix(self):
        """Resets the confusion matrix to an empty state."""
        self.confusion_matrix = torch.zeros((0, 0), dtype=torch.int64)

    def pop_confusion_matrix(self) -> np.ndarray:
        """
        Returns the accumulated confusion matrix and resets it to empty.
        
        Returns:
            The accumulated confusion matrix as a 2D numpy array.
        """
        cm = self.confusion_matrix.cpu().numpy()
        self.reset_confusion_matrix()
        return cm