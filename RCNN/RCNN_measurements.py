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
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from colorama import Fore
from colorama import Style


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
    checkpoint = torch.load("RCNN_meas/Checkpoints/rcnn_weights_checkpoint_0_2.pth",
                            map_location=device)  # Change to 'cuda' if using GPU
    model.load_state_dict(checkpoint)

    model.eval()  # Set model to evaluation mode
    h5_images_train = "/home/tauproj12/PycharmProjects/Glioma-segmentation-project/Data/train_data.h5"
    h5_masks_train = "/home/tauproj12/PycharmProjects/Glioma-segmentation-project/Data/train_masks.h5"
    h5_images_val = "/home/tauproj12/PycharmProjects/Glioma-segmentation-project/Data/val_data.h5"
    h5_masks_val = "/home/tauproj12/PycharmProjects/Glioma-segmentation-project/Data/val_masks.h5"

    h5_boxs_train = "/home/tauproj12/PycharmProjects/Glioma-segmentation-project/Data/train_boxs.h5"
    h5_boxs_val = "/home/tauproj12/PycharmProjects/Glioma-segmentation-project/Data/val_boxs.h5"

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

                mask = mask[:, :, :, 1]

                image = np.transpose(img,(2, 0, 1))  # Shape: (128, 156, 128) - reorder so index 0 will be the layer (instead of 2)
                image = np.expand_dims(image, axis=1)  # Shape: (128, 1, 128, 156)
                image = torch.tensor(image, dtype=torch.float32)
                images = list(image.unbind(0))

                boxs_pred = np.empty((0, 4))
                chunk_size = 1
                for c in range(0, len(images), chunk_size):
                    images_chunk = images[c:c+chunk_size]
                    images_chunk_ = [img.to(device) for img in images_chunk]
                    predictions = model(images_chunk_)
                    bbox = predictions[0]['boxes'].cpu().detach().numpy()
                    mask_chunk = mask[:,:,c:c+chunk_size]
                    for i in range(len(images_chunk)):
                        mask_ci = mask_chunk[:, :,i]
                        rows, cols = np.where(mask_ci > 0)
                        target_mask = 0
                        if rows.size > 0 and cols.size > 0:
                            target_mask = 1
                            x_min_target, x_max_target = cols.min(), cols.max()
                            y_min_target, y_max_target = rows.min(), rows.max()
                        else:
                            target_mask = 0

                        pred_mask = 0
                        if len(predictions[0]['labels']) == 0:
                            pred_mask = 0
                            boxs_pred = np.vstack([boxs_pred, np.array([[0, 0, 0, 0]])])
                        else:
                            if (predictions[0]['scores'][0]< 0.75):
                                pred_mask = 0
                                boxs_pred = np.vstack([boxs_pred, np.array([[0, 0, 0, 0]])])
                            else:
                                pred_mask = 1
                                x_min_pred, y_min_pred, x_max_pred, y_max_pred =  np.round(bbox[0]).astype(int)
                                boxs_pred = np.vstack([boxs_pred, np.array([[x_min_pred, y_min_pred, x_max_pred, y_max_pred]])])

                        if target_mask == 1 and pred_mask == 1:
                            pred_tensor = torch.tensor([x_min_pred, y_min_pred, x_max_pred, y_max_pred]).reshape(1, 4)
                            target_tensor = torch.tensor([x_min_target, y_min_target, x_max_target, y_max_target]).reshape(1, 4)
                            tp += 1
                            diou = compute_diou(pred_tensor,target_tensor)

                            diou_sum += diou
                        elif(target_mask == 0 and pred_mask == 0):
                            tn += 1
                        elif(target_mask == 1 and pred_mask == 0):
                            fn += 1
                        elif(target_mask == 0 and pred_mask == 1):
                            fp += 1
                #save prediction
                f_boxs.create_dataset(f"boxs_{real_idx}", data=boxs_pred)

    diou_avg = diou_sum / tp
    total = tp+tn+fp+fn
    print("train summery:")
    print(f"total slices = {total}")
    print(f"diou average = {diou_avg}")
    print("-------------------")
    print("confusion matrix:")
    print(f"{tp/total:.2f} | {fp/total:.2f}")
    print(f"{fn/total:.2f} | {tn/total:.2f}")
    print("-------------------")


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
                print(f"processing idx {idx}")
                img = img5f[image_keys[idx]]
                mask = mask5f[mask_keys[idx]]
                real_idx = image_keys[idx].split('_')[1]

                mask = mask[:, :, :, 1]

                image = np.transpose(img, (
                2, 0, 1))  # Shape: (128, 156, 128) - reorder so index 0 will be the layer (instead of 2)
                image = np.expand_dims(image, axis=1)  # Shape: (128, 1, 128, 156)
                image = torch.tensor(image, dtype=torch.float32)
                images = list(image.unbind(0))
                boxs_pred = np.empty((0, 4))
                chunk_size = 1
                for c in range(0, len(images), chunk_size):
                    images_chunk = images[c:c + chunk_size]
                    images_chunk_ = [img.to(device) for img in images_chunk]
                    predictions = model(images_chunk_)
                    bbox = predictions[0]['boxes'].cpu().detach().numpy()
                    mask_chunk = mask[:, :, c:c + chunk_size]
                    for i in range(len(images_chunk)):
                        mask_ci = mask_chunk[:, :, i]
                        rows, cols = np.where(mask_ci > 0)
                        target_mask = 0
                        if rows.size > 0 and cols.size > 0:
                            target_mask = 1
                            x_min_target, x_max_target = cols.min(), cols.max()
                            y_min_target, y_max_target = rows.min(), rows.max()
                        else:
                            target_mask = 0

                        pred_mask = 0
                        if len(predictions[0]['labels']) == 0:
                            pred_mask = 0
                            boxs_pred = np.vstack([boxs_pred, np.array([[0, 0, 0, 0]])])
                        else:
                            if (predictions[0]['scores'][0] < 0.75):
                                pred_mask = 0
                                boxs_pred = np.vstack([boxs_pred, np.array([[0, 0, 0, 0]])])
                            else:
                                pred_mask = 1
                                x_min_pred, y_min_pred, x_max_pred, y_max_pred = np.round(bbox[0]).astype(int)
                                boxs_pred = np.vstack(
                                    [boxs_pred, np.array([[x_min_pred, y_min_pred, x_max_pred, y_max_pred]])])

                        if target_mask == 1 and pred_mask == 1:
                            pred_tensor = torch.tensor([x_min_pred, y_min_pred, x_max_pred, y_max_pred]).reshape(1, 4)
                            target_tensor = torch.tensor(
                                [x_min_target, y_min_target, x_max_target, y_max_target]).reshape(1, 4)
                            tp += 1
                            diou = compute_diou(pred_tensor, target_tensor)

                            diou_sum += diou
                        elif (target_mask == 0 and pred_mask == 0):
                            tn += 1
                        elif (target_mask == 1 and pred_mask == 0):
                            fn += 1
                        elif (target_mask == 0 and pred_mask == 1):
                            fp += 1
                # save prediction
                f_boxs.create_dataset(f"boxs_{real_idx}", data=boxs_pred)

    diou_avg = diou_sum / tp
    total = tp + tn + fp + fn
    print("validation summery:")
    print(f"total slices = {total}")
    print(f"diou average = {diou_avg}")
    print("-------------------")
    print("confusion matrix:")
    print(f"{tp / total:.2f} | {fp / total:.2f}")
    print(f"{fn / total:.2f} | {tn / total:.2f}")
    print("-------------------")
