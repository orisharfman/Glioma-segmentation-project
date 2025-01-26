import torch
import torch.nn as nn

class CombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        """
        Initialize the combined loss function.
        Args:
            alpha (float): Weight for bounding box loss.
            beta (float): Weight for objectness loss.
        """
        super(CombinedLoss, self).__init__()
        self.bbox_loss_fn = nn.MSELoss(reduction='none')  # Bounding box loss
        self.obj_loss_fn = nn.BCEWithLogitsLoss()  # Objectness loss
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        """
        Compute the combined loss.
        Args:
            pred (Tensor): Predicted output of shape (batch_size, 5) -> [x_min, y_min, x_max, y_max, objectness].
            target (Tensor): Ground truth of shape (batch_size, 5) -> [x_min, y_min, x_max, y_max, objectness].
        Returns:
            Tensor: Total loss (scalar).
        """
        # Split predictions and targets
        pred_boxes, pred_obj = pred[:, :4], pred[:, 4]
        target_boxes, target_obj = target[:, :4], target[:, 4]

        # Mask for valid boxes (objectness = 1)
        mask = (target_obj == 1)

        # Compute bounding box loss (only for valid boxes)
        bbox_loss = self.bbox_loss_fn(pred_boxes, target_boxes)
        bbox_loss = bbox_loss[mask].mean() if mask.any() else torch.tensor(0.0, requires_grad=True)

        # Compute objectness loss
        obj_loss = self.obj_loss_fn(pred_obj, target_obj)

        # Combine losses
        total_loss = self.alpha * bbox_loss + self.beta * obj_loss
        return total_loss