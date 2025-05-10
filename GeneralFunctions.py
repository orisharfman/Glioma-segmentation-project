import torch
from datetime import datetime
from torchvision.ops import generalized_box_iou
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

def TimestampToString():
    current_timestamp = datetime.now()
    formatted_timestamp = current_timestamp.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_timestamp

def accuracy_calc(pred, target):
    pred_boxes, pred_obj = pred[:, :4], pred[:, 4]
    target_boxes, target_obj = target[:, :4], target[:, 4]

    true_positive = len(pred_obj[(target_obj == 1).bool() & (pred_obj>0.5).bool()])
    false_positive = len(pred_obj[(target_obj == 0).bool() & (pred_obj>0.5).bool()])
    true_negative = len(pred_obj[(target_obj == 0).bool() & (pred_obj<0.5).bool()])
    false_negative = len(pred_obj[(target_obj == 1).bool() & (pred_obj<0.5).bool()])

    hit_pred_boxes = pred_boxes[(target_obj == 1).bool() & (pred_obj>0.5).bool(),:]
    hit_target_boxes = target_boxes[(target_obj == 1).bool() & (pred_obj>0.5).bool(),:]

    if true_positive > 0:
        iou = calc_iou(hit_pred_boxes, hit_target_boxes).sum().item()
    else:
        iou = 0.0


    return [true_positive, false_positive, false_negative, true_negative, iou]

def calc_iou(pred_boxes, target_boxes):

    x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])  # Top-left x
    y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])  # Top-left y
    x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])  # Bottom-right x
    y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])  # Bottom-right y

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    pred_boxes_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    target_boxes_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])

    # Compute union area
    union = pred_boxes_area + target_boxes_area - intersection

    # Compute IoU
    iou = intersection / union

    return iou

def create_plots(path_to_train_loss, path_to_validate_loss, path_to_train_iou, path_to_validate_iou, path_to_train_dice, path_to_validate_dice):
    # path_to_train_loss = "Stats2/train_loss_arr.txt"
    # path_to_validate_loss = "Stats2/val_loss_arr.txt"
    # path_to_train_iou = "Stats2/train_iou_arr.txt"
    # path_to_validate_iou = "Stats2/validate_iou_arr.txt"

    with open(path_to_train_loss) as train_loss_file:
        train_loss_arr = np.array([float(line.strip()) for line in train_loss_file])
        train_loss_arr = train_loss_arr.reshape((train_loss_arr.shape[0]))
        print(f"train_loss_arr shape: {train_loss_arr.shape}")
    with open(path_to_validate_loss) as validate_loss_file:
        validate_loss_arr = np.array([float(line.strip()) for line in validate_loss_file])
        print(f"validate_loss_arr shape: {validate_loss_arr.shape}")
    with open(path_to_train_iou) as train_iou_file:
        train_iou_arr = [float(line.strip()) for line in train_iou_file]
    with open(path_to_validate_iou) as validate_iou_file:
        validate_iou_arr = [float(line.strip()) for line in validate_iou_file]

    with open(path_to_train_dice) as train_dice_file:
        train_dice_arr = [float(line.strip()) for line in train_dice_file]
    with open(path_to_validate_dice) as validate_dice_file:
        validate_dice_arr = [float(line.strip()) for line in validate_dice_file]

    fig, axs = plt.subplots(1, 3, figsize=(10, 8))

    axs[0].plot(range(len(train_loss_arr)),train_loss_arr, label = "Train Loss")
    axs[0].plot(range(len(validate_loss_arr)),validate_loss_arr, label = "Validation Loss")
    axs[0].set_title('Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_ylim(0, max(max(train_loss_arr), max(validate_loss_arr)) * 1.1)
    axs[0].legend()


    axs[1].plot(range(len(train_iou_arr)),train_iou_arr, label = "Train IoU")
    axs[1].plot(range(len(validate_iou_arr)),validate_iou_arr, label = "Validation IoU")
    axs[1].set_title('IoU')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('IoU')
    axs[1].yaxis.set_major_locator(MaxNLocator(nbins=5))
    axs[1].set_ylim(0, max(max(train_iou_arr), max(validate_iou_arr)) * 1.1)
    axs[1].legend()

    axs[2].plot(range(len(train_dice_arr)),train_dice_arr, label = "Train Dice")
    axs[2].plot(range(len(validate_dice_arr)),validate_dice_arr, label = "Validation Dice")
    axs[2].set_title('Dice')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Dice')
    axs[2].yaxis.set_major_locator(MaxNLocator(nbins=5))
    axs[2].set_ylim(0, max(max(train_dice_arr), max(validate_dice_arr)) * 1.1)

    axs[2].legend()



    #plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_iou.png')
    plt.show()


# path_to_train_loss = "SAM/meas_merged/loss_arr_train.txt"
# path_to_validate_loss = "SAM/meas_merged/loss_arr_val.txt"
# path_to_train_iou = "SAM/meas_merged/iou_arr_train.txt"
# path_to_validate_iou = "SAM/meas_merged/iou_arr_val.txt"
# path_to_train_dice = "SAM/meas_merged/dice_arr_train.txt"
# path_to_validate_dice = "SAM/meas_merged/dice_arr_val.txt"
# create_plots(path_to_train_loss, path_to_validate_loss, path_to_train_iou, path_to_validate_iou, path_to_train_dice, path_to_validate_dice)