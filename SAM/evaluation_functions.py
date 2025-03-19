import numpy as np
from sklearn.metrics import confusion_matrix

# sam output is a soft mask with floating point values between 0 and 1,
# needs to be converted into a binary mask for analyzing purposes.

def iou(prediction, target, threshold=0.5):
    predicted_mask = (prediction > threshold).astype(np.uint8)
    intersection = np.logical_and(target, predicted_mask).sum()
    union = np.logical_or(target, predicted_mask).sum()
    return intersection / union if union != 0 else 0

def dice(prediction, target, threshold=0.5):
    predicted_mask = (prediction > threshold).astype(np.uint8)
    intersection = np.logical_and(target, predicted_mask).sum()
    dice_score = (2 * intersection) / (target.sum() + predicted_mask.sum())
    return dice_score if (target.sum() + predicted_mask.sum()) != 0 else 1


def confusion_matrix_calculation(prediction, target, threshold=0.5):
    predicted_mask = (prediction > threshold).astype(np.uint8)
    target_flat = target.flatten()
    prediction_flat = predicted_mask.flatten()
    tn, fp, fn, tp = confusion_matrix(target_flat, prediction_flat).ravel()
    return tn, fp, fn, tp


def recall(tp, fn):
    return tp/(tp+fn)


def precision(tp, fp):
    return tp/(tp+fp)