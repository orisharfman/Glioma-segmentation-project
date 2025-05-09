import torch
import torchvision

import torchvision.transforms as transforms

import torch.nn as nn
import torch.optim as optim

import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from skimage.transform import resize
from google.protobuf.internal.well_known_types import Timestamp
from GeneralFunctions import TimestampToString

def prepare_images(images):

    images = resize(images, (300, 300, 128), anti_aliasing=True, mode="reflect")
    images = np.transpose(images, (
        2, 0, 1))  # Shape: (128, 300, 300) - reorder so index 0 will be the layer (instead of 2)
    images = np.expand_dims(images, axis=1)  # Shape: (128, 1, 300, 300)
    images = torch.tensor(images, dtype=torch.float32)
    images = list(images.unbind(0))
    return images


def prepare_targrets(masks):
    masks = resize(masks, (300, 300, 128), anti_aliasing=True, mode="reflect")
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

                # potentially add a bit of padding
                boxes.append([min_col, min_row, max_col, max_row])
                areas.append((max_col - min_col) * (max_row - min_row))
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
                if width * height < 20:  # adjust threshold as needed
                    # print("box is too small")
                    continue
                targets.append({
                    "boxes": torch.tensor(boxes, dtype=torch.float32),
                    "labels": torch.ones((len(boxes),), dtype=torch.int64),
                    "area": torch.tensor(areas, dtype=torch.float32),
                    "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
                })
        else:
            # If the slice has no non-zero elements, add empty list
            targets.append({
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.empty((0,), dtype=torch.int64),
                "area": torch.empty((0,), dtype=torch.float32),
                "iscrowd": torch.empty((0,), dtype=torch.int64),
            })
    return targets

if __name__ == "__main__":


    ssd_model = torchvision.models.detection.ssd300_vgg16(pretrained=True)

    backbone = ssd_model.backbone
    for name, module in backbone.named_children():
        if isinstance(module, nn.ReLU):  # Add dropout after ReLU layers
            dropout = nn.Dropout(p=0.5, inplace=False)
            setattr(backbone, name + "_dropout", dropout)

    ssd_model.backbone = backbone

    optimizer = optim.Adam(ssd_model.parameters(), lr=0.000001)



    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    ssd_model.to(device)



    h5_data_train = "/home/tauproj12/PycharmProjects/Glioma-segmentation-project/Data/train_data.h5"
    h5_masks_train = "/home/tauproj12/PycharmProjects/Glioma-segmentation-project/Data/train_masks.h5"
    h5_data_val = "/home/tauproj12/PycharmProjects/Glioma-segmentation-project/Data/val_data.h5"
    h5_masks_val = "/home/tauproj12/PycharmProjects/Glioma-segmentation-project/Data/val_masks.h5"
    h5_data_test = "/home/tauproj12/PycharmProjects/Glioma-segmentation-project/Data/test_data.h5"
    h5_masks_test = "/home/tauproj12/PycharmProjects/Glioma-segmentation-project/Data/test_masks.h5"

    train_loss_arr = []
    val_loss_arr = []
    num_epochs = 60

    f_image_train = h5py.File(h5_data_train, "r")
    f_masks_train = h5py.File(h5_masks_train, "r")
    f_image_val = h5py.File(h5_data_val, "r")
    f_masks_val = h5py.File(h5_masks_val, "r")
    image_train_keys = sorted(list(f_image_train.keys()))
    masks_train_keys = sorted(list(f_masks_train.keys()))
    image_val_keys = sorted(list(f_image_val.keys()))
    masks_val_keys = sorted(list(f_masks_val.keys()))
    train_size = len(image_train_keys)
    val_size = len(image_val_keys)

    chunk_size = 32

    print(f"{TimestampToString()}: start training, train_size = {train_size}, val_size = {val_size}")
    for epoch in range(num_epochs):
        ssd_model.train()
        train_loss = 0
        for key_idx in range(len(image_train_keys)):
            if image_train_keys[key_idx].split("_")[1] != masks_train_keys[key_idx].split("_")[1]:  # sanity for keys
                print(f"not matching keys! {image_train_keys[key_idx]}~{masks_train_keys[key_idx]}")
                exit(-1)
            images = f_image_train[image_train_keys[key_idx]]
            images = prepare_images(images)
            masks = f_masks_train[masks_train_keys[key_idx]]  # Shape: (128, 156, 128, 2)
            masks = masks[:, :, :, 1].astype(np.uint8) + masks[:, :, :, 2].astype(np.uint8) # Shape: (128, 156, 128)
            targets = prepare_targrets(masks)

            for i in range(0, len(images), chunk_size):
                images_chunk = images[i:i + chunk_size]
                targets_chunk = targets[i:i + chunk_size]
                total_boxes_sum = torch.zeros(4, dtype=torch.float32)
                for target in targets_chunk:
                    if target['boxes'].shape[0] > 0:
                        total_boxes_sum += target['boxes'].sum(dim=0)
                if torch.all(total_boxes_sum == 0):  # If the sum of the mask is zero, it's an empty mask
                    # print(f"TRAIN chunk {i} from image {image_keys[key_idx]} because its mask is empty.")
                    continue  # Skip this image
                images_chunk = [img.to(device) for img in images_chunk]
                targets_chunk = [{k: v.to(device) for k, v in t.items()} for t in targets_chunk]
                optimizer.zero_grad()
                loss_dict = ssd_model(images_chunk, targets_chunk)
                loss = sum(loss for loss in loss_dict.values())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ssd_model.parameters(), max_norm=1.0)  # figure out what this is doing
                optimizer.step()
                train_loss += loss.item()

        print(f"{TimestampToString()}: validation")
        val_loss = 0
        with torch.no_grad():
            for key_idx in range(len(image_val_keys)):
                if masks_val_keys[key_idx].split("_")[1] != image_val_keys[key_idx].split("_")[1]:  # sanity for keys
                    print(f"not matching keys! {masks_val_keys[key_idx]}~{image_val_keys[key_idx]}")
                    exit(-1)
                images = f_image_val[image_val_keys[key_idx]]
                images = prepare_images(images)
                masks = f_masks_val[masks_val_keys[key_idx]]  # Shape: (128, 156, 128, 2)
                masks = masks[:, :, :, 1].astype(np.uint8)  # Shape: (128, 156, 128)
                targets = prepare_targrets(masks)
                for i in range(0, len(images), chunk_size):
                    images_chunk = images[i:i + chunk_size]
                    targets_chunk = targets[i:i + chunk_size]
                    total_boxes_sum = torch.zeros(4, dtype=torch.float32)
                    for target in targets_chunk:
                        total_boxes_sum += target['boxes'].sum(dim=0)
                    if torch.all(total_boxes_sum == 0):  # If the sum of the mask is zero, it's an empty mask
                        # print(f"VAL chunk {i} from image {image_keys[key_idx]} because its mask is empty.")
                        continue  # Skip this image
                    images_chunk = [img.to(device) for img in images_chunk]
                    targets_chunk = [{k: v.to(device) for k, v in t.items()} for t in targets_chunk]
                    loss_dict = ssd_model(images_chunk, targets_chunk)  # Get the loss dictionary
                    loss = sum(loss for loss in loss_dict.values())
                    val_loss += loss.item()  # Total loss for this batch
        train_loss = train_loss / train_size / (128/chunk_size)
        val_loss = val_loss / val_size / (128/chunk_size)
        train_loss_arr.append(train_loss)
        val_loss_arr.append(val_loss)
        print(
            f"{TimestampToString()}: Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        if True:  # save checkpoint
            print(f"{TimestampToString()}: saving checkpoint on epoch {epoch + 1}")
            torch.save(ssd_model.state_dict(), f"SSD_checkpoints/SSD_weights_checkpoint_1_{int(epoch + 1)}.pth")
            np.savetxt("SSD_stats/train_loss_arr_1.txt", train_loss_arr)
            np.savetxt("SSD_stats/val_loss_arr_1.txt", val_loss_arr)