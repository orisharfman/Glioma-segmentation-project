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

def compute_diou(pred_boxes, target_boxes):
    # IoU part
    x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

    intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    area_pred = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    area_target = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
    union = area_pred + area_target - intersection

    iou = intersection / union

    # Center distance
    pred_center_x = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    pred_center_y = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
    target_center_x = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
    target_center_y = (target_boxes[:, 1] + target_boxes[:, 3]) / 2

    center_dist_sq = (pred_center_x - target_center_x) ** 2 + (pred_center_y - target_center_y) ** 2

    # Enclosing box
    enclosing_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    enclosing_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    enclosing_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    enclosing_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])

    enclosing_diag_sq = (enclosing_x2 - enclosing_x1) ** 2 + (enclosing_y2 - enclosing_y1) ** 2 + 1e-7

    # DIoU
    diou = iou - center_dist_sq / enclosing_diag_sq

    return diou

def create_plots(path_to_train_loss, path_to_validate_loss, save_path): #path_to_train_iou, path_to_validate_iou,
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
    # with open(path_to_train_iou) as train_iou_file:
    #     train_iou_arr = [line.strip() for line in train_iou_file]
    # with open(path_to_validate_iou) as validate_iou_file:
    #     validate_iou_arr = [line.strip() for line in validate_iou_file]

    # fig, axs = plt.subplots(1, 2, figsize=(10, 8))
    #
    # axs[0].plot(range(len(train_loss_arr)),train_loss_arr)
    # axs[0].set_title('Train Loss')
    # axs[0].set_xlabel('Epoch')
    # axs[0].set_ylabel('Loss')
    #
    # axs[1].plot(range(len(validate_loss_arr)),validate_loss_arr)
    # axs[1].set_title('Validation Loss')
    # axs[1].set_xlabel('Epoch')
    # axs[1].set_ylabel('Loss')

    # axs[1, 0].plot(range(len(train_iou_arr)),train_iou_arr)
    # axs[1, 0].set_title('Train IoU')
    # axs[1, 0].set_xlabel('Epoch')
    # axs[1, 0].set_ylabel('IoU')
    # axs[1, 0].yaxis.set_major_locator(MaxNLocator(nbins=5))
    #
    # axs[1, 1].plot(range(len(validate_iou_arr)),validate_iou_arr)
    # axs[1, 1].set_title('Validation IoU')
    # axs[1, 1].set_xlabel('Epoch')
    # axs[1, 1].set_ylabel('IoU')
    # axs[1, 1].yaxis.set_major_locator(MaxNLocator(nbins=5))


    #plt.grid(True)

    plt.figure(figsize=(8, 6))
    plt.plot(range(len(train_loss_arr)), train_loss_arr, label='Train Loss')
    plt.plot(range(len(validate_loss_arr)), validate_loss_arr, label='Validation Loss')

    plt.title('Train vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print("loss fig saved")