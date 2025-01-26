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
        self.bbox_loss_fn = nn.L1Loss(reduction='none')  # L1 Loss for bounding box
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
        bbox_loss = bbox_loss[mask].mean() if mask.any() else torch.tensor(0.0, device=pred.device) + 1e-6

        # Compute objectness loss
        obj_loss = self.obj_loss_fn(pred_obj, target_obj)

        # Combine losses
        total_loss = self.alpha * bbox_loss + self.beta * obj_loss
        return total_loss


import torch
import torch.nn as nn

class SimpleCombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        """
        Simplified combined loss: L1 loss for bounding box + BCE loss for objectness.
        Args:
            alpha (float): Weight for bounding box loss.
            beta (float): Weight for objectness loss.
        """
        super(SimpleCombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.bbox_loss_fn = nn.L1Loss()  # L1 loss for bounding box
        self.obj_loss_fn = nn.BCEWithLogitsLoss()  # BCE loss for objectness

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

        # Bounding box loss (L1)
        bbox_loss = self.bbox_loss_fn(pred_boxes, target_boxes)

        # Objectness loss (BCE)
        obj_loss = self.obj_loss_fn(pred_obj, target_obj)

        # Combined loss
        total_loss = self.alpha * bbox_loss + self.beta * obj_loss
        return total_loss


class MaskedCombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        """
        Masked combined loss: L1 loss for bounding box (only when obj=1) + BCE loss for objectness.
        Args:
            alpha (float): Weight for bounding box loss.
            beta (float): Weight for objectness loss.
        """
        super(MaskedCombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.bbox_loss_fn = nn.L1Loss(reduction='none')  # L1 loss for bounding box
        self.obj_loss_fn = nn.BCEWithLogitsLoss()  # BCE loss for objectness

    def forward(self, pred, target):
        """
        Compute the masked combined loss.
        Args:
            pred (Tensor): Predicted output of shape (batch_size, 5) -> [x_min, y_min, x_max, y_max, objectness].
            target (Tensor): Ground truth of shape (batch_size, 5) -> [x_min, y_min, x_max, y_max, objectness].
        Returns:
            Tensor: Total loss (scalar).
        """
        # Split predictions and targets
        pred_boxes, pred_obj = pred[:, :4], pred[:, 4]
        target_boxes, target_obj = target[:, :4], target[:, 4]

        # Create a mask for valid boxes (objectness = 1)
        mask = (target_obj == 1)

        # Compute bounding box loss (only for valid boxes)
        if mask.any():
            bbox_loss = self.bbox_loss_fn(pred_boxes[mask], target_boxes[mask]).mean()
        else:
            bbox_loss = torch.tensor(0.0, device=pred.device)

        # Compute objectness loss
        obj_loss = self.obj_loss_fn(pred_obj, target_obj)

        # Combine losses with weights
        total_loss = self.alpha * bbox_loss + self.beta * obj_loss
        return total_loss