import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torchvision
from torchvision import  transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch import no_grad
from google.protobuf.internal.well_known_types import Timestamp
from GeneralFunctions import TimestampToString , compute_diou, calc_iou


import h5py
import numpy as np
from skimage.measure import label, regionprops
import json

from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from colorama import Fore
from colorama import Style

def prepare_targrets(masks):
    targets = []  # input as list of dictionaries

    for z in range(masks.shape[2]):  # for each slice
        slice_2d = masks[:, :, z]
        # rows, cols = np.where(slice_2d > 0)
        # if rows.size > 0 and cols.size > 0:
        #     x_min, x_max = cols.min() - 3, cols.max() + 3
        #     y_min, y_max = rows.min() - 3, rows.max() + 3

        labeled = label(slice_2d)
        props = regionprops(labeled)

        if len(props) > 0:
            boxes = []
            areas = []
            for prop in props:
                min_row, min_col, max_row, max_col = prop.bbox

                #potentially add a bit of padding
                boxes.append([min_col, min_row, max_col, max_row])
                areas.append((max_col-min_col)*(max_row-min_row))
                # print(f"slice {z} box - {min_col, min_row, max_col, max_row}")
                min_col = max(0, min_col)
                max_col = min(slice_2d.shape[1], max_col)
                min_row = max(0, min_row)
                max_row = min(slice_2d.shape[0], max_row)
                width = max_col - min_col
                height = max_row - min_row

                # Skip boxes with invalid size
                if width <= 1 or height <= 1:
                    # print(f"Bad box in slice {z}: {min_row, min_col, max_row, max_col}")
                    continue

                # Optional: skip tiny regions
                if width * height < 10:  # adjust threshold as needed
                    # print("box is too small")
                    continue
            targets.append({
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.ones((len(boxes),), dtype=torch.int64),
                "image_id": torch.tensor([z]),
                "area": torch.tensor(areas, dtype=torch.float32),
                "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            })
        else:
            # If the slice has no non-zero elements, add empty list
            targets.append({
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.empty((0,), dtype=torch.int64),
                "image_id": torch.tensor([z]),
                "area": torch.empty((0,), dtype=torch.float32),
                "iscrowd": torch.empty((0,), dtype=torch.int64),
            })
    return targets



if __name__ == "__main__":

    torch.cuda.empty_cache()  # Clears unused memory from cache
    torch.cuda.ipc_collect()  # Forces garbage collection of unused tensors

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    diou_sum = 0

    num_classes = 2  # only 1 class to find, and additional for background.
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)


    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    checkpoint = torch.load("RCNN_meas/Checkpoints/rcnn_weights_checkpoint_19_15.pth",
                            map_location=device)  # Change to 'cuda' if using GPU
    model.load_state_dict(checkpoint)

    model.eval()  # Set model to evaluation mode
    h5_images_train = "../Data/train_data.h5"
    h5_masks_train = "../Data/train_masks.h5"
    h5_images_val = "../Data/val_data.h5"
    h5_masks_val = "../Data/val_masks.h5"

    h5_boxs_train = "../Data/train_boxs.h5"
    h5_boxs_val = "../Data/val_boxs.h5"
    os.makedirs("RCNN_meas/Stats", exist_ok=True)  # Creates the directory if needed

    color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    with h5py.File(h5_masks_train, 'r') as mask5f:
        mask_keys = [key for key in mask5f.keys()]
        mask_keys.sort()
        with h5py.File(h5_images_train, 'r') as img5f:
            f_boxs = h5py.File(h5_boxs_train, "w")
            image_keys = [key for key in img5f.keys()]
            image_keys.sort()
            for idx in range(len(image_keys)):
                print(f"processing idx {idx}")
                img = img5f[image_keys[idx]]
                mask = mask5f[mask_keys[idx]]
                real_idx = image_keys[idx].split('_')[1]

                mask = mask[:, :, :, 1] + mask[:, :, :, 2]

                image = np.transpose(img,(2, 0, 1))  # Shape: (128, 156, 128) - reorder so index 0 will be the layer (instead of 2)
                image = np.expand_dims(image, axis=1)  # Shape: (128, 1, 128, 156)
                image = torch.tensor(image, dtype=torch.float32)
                images = list(image.unbind(0))

                boxs_pred = np.empty((0, 4))
                chunk_size = 1
                scores_per_slice = {}  # To store all prediction scores per slice

                for c in range(0, len(images), chunk_size):
                    images_chunk = images[c:c + chunk_size]
                    images_chunk_ = [img.to(device) for img in images_chunk]

                    with torch.no_grad():
                        predictions = model(images_chunk_)

                    # Prepare all targets for this chunk
                    mask_chunk = mask[:, :, c:c + chunk_size]
                    targets_chunk = prepare_targrets(mask_chunk)

                    for i, (pred, target) in enumerate(zip(predictions, targets_chunk)):
                        slice_idx = f"slice_{real_idx}_{c + i}"

                        pred_boxes = pred['boxes'].cpu()
                        pred_scores = pred['scores'].cpu()
                        pred_labels = pred['labels'].cpu()

                        target_boxes = target['boxes']
                        target_labels = target['labels']

                        # Save all raw prediction scores per slice
                        scores_per_slice[slice_idx] = pred_scores.tolist()

                        # Check for any valid predictions using score threshold only
                        has_prediction = (pred_scores > 0.75).any().item()
                        has_target = len(target_boxes) > 0

                        # Add predicted boxes to output (filtered for visualization/statistics)
                        keep = pred_scores > 0.75
                        filtered_pred_boxes = pred_boxes[keep]
                        boxs_pred = np.vstack([
                            boxs_pred,
                            filtered_pred_boxes.detach().numpy() if len(filtered_pred_boxes) > 0 else np.array(
                                [[0, 0, 0, 0]])
                        ])
                        if len(filtered_pred_boxes) == 0 and len(target_boxes) == 0:
                            tn += 1
                            # print("this slice is TN")
                        elif len(filtered_pred_boxes) == 0 and len(target_boxes) > 0:
                            fn += 1
                            # print("this slice is FN")
                        elif len(filtered_pred_boxes) > 0 and len(target_boxes) == 0:
                            fp += 1
                            # print("this slice is FP")
                        else:
                            # both have predictions and targets
                            # match preds and targets based on IoU
                            ious = torch.zeros((len(filtered_pred_boxes), len(target_boxes)))

                            for pred_idx in range(len(filtered_pred_boxes)):
                                for target_idx in range(len(target_boxes)):
                                    ious[pred_idx, target_idx] = compute_diou(
                                        filtered_pred_boxes[pred_idx].unsqueeze(0),
                                        target_boxes[target_idx].unsqueeze(0)
                                    )
                            matched_pred = set()
                            matched_target = set()

                            for pred_idx in range(ious.shape[0]):
                                max_iou, target_idx = ious[pred_idx].max(0)
                                if pred_idx >= len(filtered_pred_boxes):
                                    break
                                if max_iou > 0.75:  # IoU threshold for considering a match
                                    matched_pred.add(pred_idx)
                                    matched_target.add(target_idx.item())
                                    diou_sum += compute_diou(
                                        filtered_pred_boxes[pred_idx].unsqueeze(0),
                                        target_boxes[target_idx].unsqueeze(0)
                                    )

                            if len(matched_pred) > 0:
                                tp += 1
                                # print("this slice is TP")
                            else:
                                fp += 1
                                # print("this slice is FP (predictions exist but no good match)")
                        # # Confusion matrix logic based on prediction *presence*, not IoU
                        # if not has_prediction and not has_target:
                        #     tn += 1
                        # elif not has_prediction and has_target:
                        #     fn += 1
                        # elif has_prediction and not has_target:
                        #     fp += 1
                        # else:
                        #     tp += 1
                        #     # Optionally compute DIoU for each matching pred-target pair above score threshold
                        #     for tb in target_boxes:
                        #         for j, pb in enumerate(pred_boxes):
                        #             if pred_scores[j] > 0.8:
                        #                 diou_sum += compute_diou(pb.unsqueeze(0), tb.unsqueeze(0))
                        #                 break  # only need one match per target for DIoU

                # Save prediction scores to JSON
                with open("RCNN_meas/Stats/train_slice_scores.json", "w") as f:
                    json.dump(scores_per_slice, f, indent=2)

                # for c in range(0, len(images), chunk_size):
                #     images_chunk = images[c:c+chunk_size]
                #     images_chunk_ = [img.to(device) for img in images_chunk]
                #     predictions = model(images_chunk_)
                #     bbox = predictions[0]['boxes'].cpu().detach().numpy()
                #     mask_chunk = mask[:,:,c:c+chunk_size]
                #     for i in range(len(images_chunk)):
                #         mask_ci = mask_chunk[:, :,i]
                #         rows, cols = np.where(mask_ci > 0)
                #         target_mask = 0
                #         if rows.size > 0 and cols.size > 0:
                #             target_mask = 1
                #             x_min_target, x_max_target = cols.min(), cols.max()
                #             y_min_target, y_max_target = rows.min(), rows.max()
                #         else:
                #             target_mask = 0
                #
                #         pred_mask = 0
                #         if len(predictions[0]['labels']) == 0:
                #             pred_mask = 0
                #             boxs_pred = np.vstack([boxs_pred, np.array([[0, 0, 0, 0]])])
                #         else:
                #             if (predictions[0]['scores'][0]< 0.75):
                #                 pred_mask = 0
                #                 boxs_pred = np.vstack([boxs_pred, np.array([[0, 0, 0, 0]])])
                #             else:
                #                 pred_mask = 1
                #                 x_min_pred, y_min_pred, x_max_pred, y_max_pred =  np.round(bbox[0]).astype(int)
                #                 boxs_pred = np.vstack([boxs_pred, np.array([[x_min_pred, y_min_pred, x_max_pred, y_max_pred]])])
                #
                #         if target_mask == 1 and pred_mask == 1:
                #             pred_tensor = torch.tensor([x_min_pred, y_min_pred, x_max_pred, y_max_pred]).reshape(1, 4)
                #             target_tensor = torch.tensor([x_min_target, y_min_target, x_max_target, y_max_target]).reshape(1, 4)
                #             tp += 1
                #             diou = compute_diou(pred_tensor,target_tensor)
                #
                #             diou_sum += diou
                #         elif(target_mask == 0 and pred_mask == 0):
                #             tn += 1
                #         elif(target_mask == 1 and pred_mask == 0):
                #             fn += 1
                #         elif(target_mask == 0 and pred_mask == 1):
                #             fp += 1
                #save prediction
                # f_boxs.create_dataset(f"boxs_{real_idx}", data=boxs_pred)

    diou_avg = diou_sum / tp
    total = tp+tn+fp+fn
    print("train summery:")
    print(f"total slices = {total}")
    print(f"diou average = {diou_avg}")
    print("-------------------")
    print("confusion matrix:")
    print(f"TP:{tp/total:.2f} | FP:{fp/total:.2f}")
    print(f"FN:{fn/total:.2f} | TN:{tn/total:.2f}")
    print("-------------------")
    # with open("RCNN_meas/Stats/train_summary.txt", "w") as f:
    #     f.write("train summary:\n")
    #     f.write(f"total slices = {total}\n")
    #     f.write(f"diou average = {diou_avg:.4f}\n")
    #     f.write("-------------------\n")
    #     f.write("confusion matrix:\n")
    #     f.write(f"{tp / total:.2f} | {fp / total:.2f}\n")
    #     f.write(f"{fn / total:.2f} | {tn / total:.2f}\n")
    #     f.write("-------------------\n")

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    diou_sum = 0

    with h5py.File(h5_masks_val, 'r') as mask5f:
        mask_keys = [key for key in mask5f.keys()]
        mask_keys.sort()

        with h5py.File(h5_images_val, 'r') as img5f:
            f_boxs = h5py.File(h5_boxs_val, "w")
            image_keys = [key for key in img5f.keys()]
            image_keys.sort()

            for idx in range(len(image_keys)):
            # for idx in range(1):

                print(f"processing idx {idx}")
                img = img5f[image_keys[idx]]
                mask = mask5f[mask_keys[idx]]
                real_idx = image_keys[idx].split('_')[1]

                mask = mask[:, :, :, 1] + mask[:, :, :, 2]

                image = np.transpose(img, (
                2, 0, 1))  # Shape: (128, 156, 128) - reorder so index 0 will be the layer (instead of 2)
                image = np.expand_dims(image, axis=1)  # Shape: (128, 1, 128, 156)
                image = torch.tensor(image, dtype=torch.float32)
                images = list(image.unbind(0))
                boxs_pred = np.empty((0, 4))
                chunk_size = 1
                # for c in range(0, len(images), chunk_size):
                #     images_chunk = images[c:c + chunk_size]
                #     images_chunk_ = [img.to(device) for img in images_chunk]
                #     predictions = model(images_chunk_)
                #     bbox = predictions[0]['boxes'].cpu().detach().numpy()
                #     mask_chunk = mask[:, :, c:c + chunk_size]
                #     for i in range(len(images_chunk)):
                #         mask_ci = mask_chunk[:, :, i]
                #         rows, cols = np.where(mask_ci > 0)
                #         target_mask = 0
                #         if rows.size > 0 and cols.size > 0:
                #             target_mask = 1
                #             x_min_target, x_max_target = cols.min(), cols.max()
                #             y_min_target, y_max_target = rows.min(), rows.max()
                #         else:
                #             target_mask = 0
                #
                #         pred_mask = 0
                #         if len(predictions[0]['labels']) == 0:
                #             pred_mask = 0
                #             boxs_pred = np.vstack([boxs_pred, np.array([[0, 0, 0, 0]])])
                #         else:
                #             if (predictions[0]['scores'][0] < 0.8):
                #                 pred_mask = 0
                #                 boxs_pred = np.vstack([boxs_pred, np.array([[0, 0, 0, 0]])])
                #             else:
                #                 pred_mask = 1
                #                 x_min_pred, y_min_pred, x_max_pred, y_max_pred = np.round(bbox[0]).astype(int)
                #                 boxs_pred = np.vstack(
                #                     [boxs_pred, np.array([[x_min_pred, y_min_pred, x_max_pred, y_max_pred]])])
                #
                #         if target_mask == 1 and pred_mask == 1:
                #             pred_tensor = torch.tensor([x_min_pred, y_min_pred, x_max_pred, y_max_pred]).reshape(1, 4)
                #             target_tensor = torch.tensor(
                #                 [x_min_target, y_min_target, x_max_target, y_max_target]).reshape(1, 4)
                #             tp += 1
                #             diou = compute_diou(pred_tensor, target_tensor)
                #
                #             diou_sum += diou
                #         elif (target_mask == 0 and pred_mask == 0):
                #             tn += 1
                #         elif (target_mask == 1 and pred_mask == 0):
                #             fn += 1
                #         elif (target_mask == 0 and pred_mask == 1):
                #             fp += 1
                scores_per_slice = {}  # To store all prediction scores per slice

                for c in range(0, len(images), chunk_size):
                    images_chunk = images[c:c + chunk_size]
                    images_chunk_ = [img.to(device) for img in images_chunk]

                    with torch.no_grad():
                        predictions = model(images_chunk_)

                    # Prepare all targets for this chunk
                    mask_chunk = mask[:, :, c:c + chunk_size]
                    targets_chunk = prepare_targrets(mask_chunk)

                    for i, (pred, target) in enumerate(zip(predictions, targets_chunk)):
                        slice_idx = f"slice_{real_idx}_{c + i}"

                        pred_boxes = pred['boxes'].cpu()
                        pred_scores = pred['scores'].cpu()
                        pred_labels = pred['labels'].cpu()

                        target_boxes = target['boxes']
                        target_labels = target['labels']

                        # Save all raw prediction scores per slice
                        scores_per_slice[slice_idx] = pred_scores.tolist()

                        # Check for any valid predictions using score threshold only
                        has_prediction = (pred_scores > 0.75).any().item()
                        has_target = len(target_boxes) > 0

                        # Add predicted boxes to output (filtered for visualization/statistics)
                        keep = pred_scores > 0.75
                        filtered_pred_boxes = pred_boxes[keep]
                        boxs_pred = np.vstack([
                            boxs_pred,
                            filtered_pred_boxes.detach().numpy() if len(filtered_pred_boxes) > 0 else np.array(
                                [[0, 0, 0, 0]])
                        ])
                        if len(filtered_pred_boxes) == 0 and len(target_boxes) == 0:
                            tn += 1
                            # print("this slice is TN")
                        elif len(filtered_pred_boxes) == 0 and len(target_boxes) > 0:
                            fn += 1
                            # print("this slice is FN")
                        elif len(filtered_pred_boxes) > 0 and len(target_boxes) == 0:
                            fp += 1
                            # print("this slice is FP")
                        else:
                            # both have predictions and targets
                            # match preds and targets based on IoU
                            ious = torch.zeros((len(filtered_pred_boxes), len(target_boxes)))

                            for pred_idx in range(len(filtered_pred_boxes)):
                                for target_idx in range(len(target_boxes)):
                                    ious[pred_idx, target_idx] = compute_diou(
                                        filtered_pred_boxes[pred_idx].unsqueeze(0),
                                        target_boxes[target_idx].unsqueeze(0)
                                    )
                            matched_pred = set()
                            matched_target = set()

                            for pred_idx in range(ious.shape[0]):
                                max_iou, target_idx = ious[pred_idx].max(0)
                                if pred_idx >= len(filtered_pred_boxes):
                                    break
                                if max_iou > 0.75:  # IoU threshold for considering a match
                                    matched_pred.add(pred_idx)
                                    matched_target.add(target_idx.item())
                                    diou_sum += compute_diou(
                                        filtered_pred_boxes[pred_idx].unsqueeze(0),
                                        target_boxes[target_idx].unsqueeze(0)
                                    )

                            if len(matched_pred) > 0:
                                tp += 1
                                # print("this slice is TP")
                            else:
                                fp += 1
                                # print("this slice is FP (predictions exist but no good match)")
                        # # Confusion matrix logic based on prediction *presence*, not IoU
                        # if not has_prediction and not has_target:
                        #     tn += 1
                        #     print("this slice is TN")
                        # elif not has_prediction and has_target:
                        #     fn += 1
                        #     print("this slice is FN")
                        # elif has_prediction and not has_target:
                        #     fp += 1
                        #     print("this slice is FP")
                        # else:
                        #     tp += 1
                        #     print("this slice is TP")
                        #     # Optionally compute DIoU for each matching pred-target pair above score threshold
                        #     for tb in target_boxes:
                        #         for j, pb in enumerate(pred_boxes):
                        #             if pred_scores[j] > 0.8:
                        #                 diou_sum += compute_diou(pb.unsqueeze(0), tb.unsqueeze(0))
                        #                 break  # only need one match per target for DIoU

                # Save prediction scores to JSON
                with open("RCNN_meas/Stats/val_slice_scores.json", "w") as f:
                    json.dump(scores_per_slice, f, indent=2)

                # save prediction

                # f_boxs.create_dataset(f"boxs_{real_idx}", data=boxs_pred)

    diou_avg = diou_sum / tp
    total = tp + tn + fp + fn
    print("validation summery:")
    print(f"total slices = {total}")
    print(f"diou average = {diou_avg}")
    print("-------------------")
    print("confusion matrix:")
    print(f"TP:{tp / total:.2f} | FP:{fp / total:.2f}")
    print(f"FN:{fn / total:.2f} | TN:{tn / total:.2f}")
    print("-------------------")
    # with open("RCNN_meas/Stats/validation_summary.txt", "w") as f:
    #     f.write("validation summary:\n")
    #     f.write(f"total slices = {total}\n")
    #     f.write(f"diou average = {diou_avg:.4f}\n")
    #     f.write("-------------------\n")
    #     f.write("confusion matrix:\n")
    #     f.write(f"{tp / total:.2f} | {fp / total:.2f}\n")
    #     f.write(f"{fn / total:.2f} | {tn / total:.2f}\n")
    #     f.write("-------------------\n")

