import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torchvision
from torchvision import  transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch import no_grad
from google.protobuf.internal.well_known_types import Timestamp
from GeneralFunctions import TimestampToString
from sklearn.preprocessing import MinMaxScaler
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.measure import label, regionprops
import albumentations as A
from albumentations.pytorch import ToTensorV2
from colorama import Fore
from colorama import Style


import torchvision.transforms.functional as TF
import random
import torch


def augment_batch(images, masks):
    H, W = images.shape[1], images.shape[0]
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=30, p=0.5),
        # A.ShiftScaleRotate(
        #     shift_limit=0.05,
        #     scale_limit=0.05,
        #     rotate_limit=15,
        #     p=0.5
        # ),
        A.PadIfNeeded(min_height=H, min_width=W, border_mode=0),
        A.Resize(height=H, width=W),
    ])
    aug_images = []
    aug_masks = []
    print(f"images shape - {images.shape}")
    print(f"masks shape - {masks.shape}")
    for i in range(images.shape[0]):
        image_slice = images[i, :, :]
        mask_slice = masks[i, :, :]

        image_slice = np.expand_dims(image_slice, axis=-1)  # (H, W, 1)


        augmented = transform(image=image_slice, mask=mask_slice)
        aug_images.append(augmented['image'].squeeze(-1))
        aug_masks.append(augmented['mask'])
    print(f"aug_images shape - {len(aug_images)}")
    print(f"aug_masks shape - {len(aug_masks)}")
    # Reconstruct augmented arrays
    aug_images = np.stack(aug_images, axis=0).reshape(images.shape)
    aug_masks = np.stack(aug_masks, axis=0).reshape(masks.shape)

    return aug_images, aug_masks

def prepare_images(images):
    images = np.transpose(images, (
        2, 0, 1))  # Shape: (128, 156, 128) - reorder so index 0 will be the layer (instead of 2)
    images = np.expand_dims(images, axis=1)  # Shape: (128, 1, 128, 156)

    # Reshape back to original shape (128, 1, 128, 156)
    images = images.reshape(128, 1, 128, 156)

    images = torch.tensor(images, dtype=torch.float32)
    images = list(images.unbind(0))
    return images


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
    print(torch.version.cuda)


    torch.cuda.empty_cache()  # Clears unused memory from cache

    torch.cuda.ipc_collect()


    num_classes = 2  # only 1 class to find, and additional for background.
    empty_chunks = 0
    empty_ratio = 0.4
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    print(f"cuda available = {torch.cuda.is_available()}")
    # device = torch.device("cpu")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # checkpoint = torch.load("RCNN_meas/Checkpoints/rcnn_weights_checkpoint_17_12.pth",
    #                         map_location=device)  # Change to 'cuda' if using GPU
    # model.load_state_dict(checkpoint)

    # #change first layer to accept 1 chunnel
    # old_conv = model.backbone.body.conv1
    # new_conv = nn.Conv2d( # Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
    #     in_channels=1,  # Change from 3 to 1
    #     out_channels=old_conv.out_channels,  # Keep same number of output channels
    #     kernel_size=old_conv.kernel_size,
    #     stride=old_conv.stride,
    #     padding=old_conv.padding,
    #     bias=False
    # )
    # model.backbone.body.conv1 = new_conv
    #
    # old_cls_logits = model.rpn.head.cls_logits
    # new_cls_logits = nn.Conv2d(  # Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
    #     in_channels=old_cls_logits.in_channels,  # Change from 3 to 1
    #     out_channels=1,  # Keep same number of output channels
    #     kernel_size=old_cls_logits.kernel_size,
    #     stride=old_cls_logits.stride,
    #     padding=old_cls_logits.padding,
    #     bias=False
    # )
    # model.rpn.head.cls_logits = new_cls_logits
    # print(model.eval())
    #
    # model.transform.mean = [0.5]  # Adjust mean for grayscale
    # model.transform.std = [0.5]  # Adjust std for grayscale
    chunk_size = 8

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    # for param in model.backbone.body.parameters():
    #     param.requires_grad = False #refrain from using the  convolutional layers in the beginning to prevent overfitting
    model.to(device)



    h5_data_train = "../Data/train_data.h5"
    h5_masks_train = "../Data/train_masks.h5"

    h5_data_val = "../Data/val_data.h5"
    h5_masks_val = "../Data/val_masks.h5"

    h5_data_test = "../Data/test_data.h5"
    h5_masks_test = "../Data/test_masks.h5"

    train_loss_arr = []
    val_loss_arr = []
    num_epochs = 30


    with h5py.File(h5_data_train, "r") as f_image, h5py.File(h5_masks_train, "r") as f_mask:
        image_keys = sorted(list(f_image.keys()))
        mask_keys = sorted(list(f_mask.keys()))
        train_size = len(image_keys)
    with h5py.File(h5_data_val, "r") as f_image, h5py.File(h5_masks_val, "r") as f_mask:
        image_val_keys = sorted(list(f_image.keys()))
        mask_val_keys = sorted(list(f_mask.keys()))
        val_size = len(image_val_keys)


    model.train()
    # print(f"{TimestampToString()}: start initial loss evaluation, train_size = {train_size}, val_size = {val_size}")
    # train_loss = 0
    # tot_num_chunks_train = 0
    # with h5py.File(h5_data_train, "r") as f_image, h5py.File(h5_masks_train, "r") as f_mask:
    #     with torch.no_grad():
    #
    #         for key_idx in range(len(image_keys)):
    #             if mask_keys[key_idx].split("_")[1] != image_keys[key_idx].split("_")[1]:  # sanity for keys
    #                 print(f"not matching keys! {mask_keys[key_idx]}~{image_keys[key_idx]}")
    #                 exit(-1)
    #             images = f_image[image_keys[key_idx]]
    #             images = prepare_images(images)
    #             masks = f_mask[mask_keys[key_idx]]  # Shape: (128, 156, 128, 2)
    #             masks = masks[:, :, :, 1].astype(np.uint8) +masks[:, :, :, 2].astype(np.uint8)
    #             targets = prepare_targrets(masks)
    #             for i in range(0, len(images), chunk_size):
    #                 images_chunk = images[i:i + chunk_size]
    #                 targets_chunk = targets[i:i + chunk_size]
    #                 total_boxes_sum = torch.zeros(4, dtype=torch.float32)
    #                 for target in targets_chunk:
    #                     if target['boxes'].shape[0] > 0:
    #                         total_boxes_sum += target['boxes'].sum(dim=0)
    #                 if torch.all(total_boxes_sum == 0):  # If the sum of the mask is zero, it's an empty mask
    #                     # print(f"TRAIN chunk {i} from image {image_keys[key_idx]} because its mask is empty.")
    #                     if random.random() > empty_ratio:
    #                         continue  # Skip this image
    #                 images_chunk = [img.to(device) for img in images_chunk]
    #                 targets_chunk = [{k: v.to(device) for k, v in t.items()} for t in targets_chunk]
    #                 loss_dict = model(images_chunk, targets_chunk)
    #                 # loss = loss_dict["loss_classifier"]*0.1+loss_dict["loss_box_reg"]*2+loss_dict["loss_objectness"]*1+loss_dict["loss_rpn_box_reg"]*2
    #                 loss = sum(loss for loss in loss_dict.values())
    #                 train_loss += loss.item()
    #                 tot_num_chunks_train += 1
    #                 # print(f"=== init TRAIN LOSSES chunk {i} ===")
    #                 # for k, v in loss_dict.items():
    #                 #     print(f"{k}: {v.item()}")
    #                 # print(f"tot_num_chunks_train - {tot_num_chunks_train}")
    #
    #
    # print(f"{TimestampToString()}: validation")
    # val_loss = 0
    # empty_chunks = 0
    # tot_num_chunks_val = 0
    # with h5py.File(h5_data_val, "r") as f_image, h5py.File(h5_masks_val, "r") as f_mask:
    #     with torch.no_grad():
    #         for key_idx in range(len(image_val_keys)):
    #             if mask_val_keys[key_idx].split("_")[1] != image_val_keys[key_idx].split("_")[1]:  # sanity for keys
    #                 print(f"not matching keys! {mask_keys[key_idx]}~{image_val_keys[key_idx]}")
    #             images = f_image[image_val_keys[key_idx]]
    #             images = prepare_images(images)
    #
    #             masks = f_mask[mask_val_keys[key_idx]]  # Shape: (128, 156, 128, 2)
    #             masks = masks[:, :, :, 1].astype(np.uint8) + masks[:, :, :, 2].astype(np.uint8)  # Shape: (128, 156, 128)
    #             targets = prepare_targrets(masks)
    #
    #             for i in range(0, len(images), chunk_size):
    #                 images_chunk = images[i:i + chunk_size]
    #                 targets_chunk = targets[i:i + chunk_size]
    #                 total_boxes_sum = torch.zeros(4, dtype=torch.float32)
    #                 for target in targets_chunk:
    #                     if target['boxes'].shape[0] > 0:
    #                         total_boxes_sum += target['boxes'].sum(dim=0)
    #                 if torch.all(total_boxes_sum == 0):  # If the sum of the mask is zero, it's an empty mask
    #                     # print(f"TRAIN chunk {i} from image {image_keys[key_idx]} because its mask is empty.")
    #                     if random.random() > empty_ratio:
    #                         continue  # Skip this image
    #                 images_chunk = [img.to(device) for img in images_chunk]
    #                 targets_chunk = [{k: v.to(device) for k, v in t.items()} for t in targets_chunk]
    #                 # Run the model on the validation images
    #                 loss_dict = model(images_chunk, targets_chunk)  # Get the loss dictionary
    #                 # loss = loss_dict["loss_classifier"]*0.1+loss_dict["loss_box_reg"]*2+loss_dict["loss_objectness"]*1+loss_dict["loss_rpn_box_reg"]*2
    #                 loss = sum(loss for loss in loss_dict.values())
    #                 val_loss += loss.item()  # Total loss for this batch
    #                 tot_num_chunks_val += 1
    #                 # print(f"=== init VAL LOSSES chunk {i} ===")
    #                 # for k, v in loss_dict.items():
    #                 #     print(f"{k}: {v.item()}")
    #                 # print(f"tot_num_chunks_val - {tot_num_chunks_val}")
    # print(f"before normalizing - tot_num_chunks_train - {tot_num_chunks_train}, tot_num_chunks_val - {tot_num_chunks_val}")
    # train_loss = train_loss / tot_num_chunks_train
    # val_loss = val_loss / tot_num_chunks_val
    #
    # print(f"{TimestampToString()}: Initial loss - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    # np.savetxt("RCNN_meas/Stats/initial_train_loss_17.txt", [train_loss])
    # np.savetxt("RCNN_meas/Stats/initial_val_loss_17.txt", [val_loss])

    print(f"{TimestampToString()}: start training, train_size = {train_size}, val_size = {val_size}")
    for epoch in range(num_epochs):
        # optimizer.zero_grad()
        model.train()
        train_loss = 0
        tot_num_chunks_train = 0
        # if epoch == 7:
        #     for param in model.backbone.body.parameters():
        #         param.requires_grad = True #use the convolutional layers once again

        with h5py.File(h5_data_train, "r") as f_image, h5py.File(h5_masks_train, "r") as f_mask:
            for key_idx in range(len(image_keys)):
                if mask_keys[key_idx].split("_")[1]!=image_keys[key_idx].split("_")[1]: #sanity for keys
                    print(f"not matching keys! {mask_keys[key_idx]}~{image_keys[key_idx]}")
                    exit(-1)
                images = np.array(f_image[image_keys[key_idx]])

                # print(f"IMAGE {image_keys[key_idx]}")
                masks = f_mask[mask_keys[key_idx]] # Shape: (128, 156, 128, 2)
                masks = masks[:, :, :, 1].astype(np.uint8) + masks[:, :, :, 2].astype(np.uint8)# Shape: (128, 156, 128)

                # images, masks = augment_batch(images, masks)

                images = prepare_images(images)
                targets = prepare_targrets(masks)
                for i in range(0,len(images),chunk_size):
                    images_chunk = images[i:i+chunk_size]
                    targets_chunk = targets[i:i+chunk_size]
                    total_boxes_sum = torch.zeros(4, dtype=torch.float32)
                    for target in targets_chunk:
                        if target['boxes'].shape[0] > 0:
                            total_boxes_sum += target['boxes'].sum(dim=0)
                    if torch.all(total_boxes_sum == 0):  # If the sum of the mask is zero, it's an empty mask
                        # print(f"TRAIN chunk {i} from image {image_keys[key_idx]} because its mask is empty.")
                        if random.random() > empty_ratio:
                            continue  # Skip this image
                    for t_idx, t in enumerate(targets_chunk):
                        boxes = t["boxes"]
                        if boxes.numel() > 0:
                            # Check for NaNs or Infs
                            if not torch.isfinite(boxes).all():
                                print(f"NaN or Inf in boxes at chunk {i}, target {t_idx}")
                                continue

                            # Check for invalid shapes
                            widths = boxes[:, 2] - boxes[:, 0]
                            heights = boxes[:, 3] - boxes[:, 1]
                            if (widths <= 0).any() or (heights <= 0).any():
                                print(f"Invalid box dimensions at chunk {i}, target {t_idx}")
                                print("Boxes:", boxes)
                                continue
                    images_chunk = [img.to(device) for img in images_chunk]
                    targets_chunk = [{k: v.to(device) for k, v in t.items()} for t in targets_chunk]
                    optimizer.zero_grad()
                    loss_dict = model(images_chunk, targets_chunk)
                    # loss = loss_dict["loss_classifier"]*0.1+loss_dict["loss_box_reg"]*2+loss_dict["loss_objectness"]*1+loss_dict["loss_rpn_box_reg"]*2
                    if loss_dict["loss_box_reg"] > 1000000:
                        print(f"loss got too big!!!! loss_dict[\"loss_box_reg\"] = {loss_dict["loss_box_reg"]}")
                        exit(0)
                    # loss = loss_dict["loss_box_reg"] + loss_dict["loss_rpn_box_reg"]
                    loss = sum(loss for loss in loss_dict.values())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #figure out what this is doing
                    optimizer.step()
                    train_loss += loss.item()
                    tot_num_chunks_train += 1
                    # print("=== TRAIN TARGETS ===")
                    # for i, t in enumerate(targets):
                    #     print(f"Image {i}:")
                    #     print("  Boxes:", t['boxes'])
                    #     print("  Labels:", t['labels'])
                    # if i%5 == 0 or loss_dict["loss_box_reg"] > 1000:
                    #     print(f"=== TRAIN LOSSES chunk {i} ===")
                    #     for k, v in loss_dict.items():
                    #         print(f"{k}: {v.item()}")
        scheduler.step()

        print(f"{TimestampToString()}: validation")
        val_loss = 0
        tot_num_chunks_val = 0

        with h5py.File(h5_data_val, "r") as f_image, h5py.File(h5_masks_val, "r") as f_mask:
            with torch.no_grad():
                for key_idx in range(len(image_val_keys)):
                    if mask_val_keys[key_idx].split("_")[1] != image_val_keys[key_idx].split("_")[1]:  # sanity for keys
                        print(f"not matching keys! {mask_keys[key_idx]}~{image_val_keys[key_idx]}")
                        exit(-1)
                    images = f_image[image_val_keys[key_idx]]
                    images = prepare_images(images)

                    masks = f_mask[mask_val_keys[key_idx]]  # Shape: (128, 156, 128, 2)
                    masks = masks[:, :, :, 1].astype(np.uint8) + masks[:, :, :, 2].astype(np.uint8) # Shape: (128, 156, 128)
                    targets = prepare_targrets(masks)


                    for i in range(0, len(images), chunk_size):
                        images_chunk = images[i:i + chunk_size]
                        targets_chunk = targets[i:i + chunk_size]
                        total_boxes_sum = torch.zeros(4, dtype=torch.float32)
                        for target in targets_chunk:
                            total_boxes_sum += target['boxes'].sum(dim=0)
                        if torch.all(total_boxes_sum == 0):  # If the sum of the mask is zero, it's an empty mask
                            # print(f"VAL chunk {i} from image {image_keys[key_idx]} because its mask is empty.")
                            if random.random() > empty_ratio:
                                continue  # Skip this image
                        for t_idx, t in enumerate(targets_chunk):
                            boxes = t["boxes"]
                            if boxes.numel() > 0:
                                # Check for NaNs or Infs
                                if not torch.isfinite(boxes).all():
                                    print(f"NaN or Inf in boxes at chunk {i}, target {t_idx}")
                                    continue

                                # Check for invalid shapes
                                widths = boxes[:, 2] - boxes[:, 0]
                                heights = boxes[:, 3] - boxes[:, 1]
                                if (widths <= 0).any() or (heights <= 0).any():
                                    print(f"Invalid box dimensions at chunk {i}, target {t_idx}")
                                    print("Boxes:", boxes)
                                    continue
                        images_chunk = [img.to(device) for img in images_chunk]
                        targets_chunk = [{k: v.to(device) for k, v in t.items()} for t in targets_chunk]
                        # Run the model on the validation images
                        loss_dict = model(images_chunk, targets_chunk)  # Get the loss dictionary
                        # loss = loss_dict["loss_classifier"] * 0.1 + loss_dict["loss_box_reg"] * 2 + loss_dict[
                        #     "loss_objectness"] * 1 + loss_dict["loss_rpn_box_reg"] * 2
                        # loss = loss_dict["loss_box_reg"] + loss_dict["loss_rpn_box_reg"]
                        loss = sum(loss for loss in loss_dict.values())
                        val_loss += loss.item()# Total loss for this batch
                        tot_num_chunks_val += 1
                        # print("=== VAL LOSSES ===")
                        # for k, v in loss_dict.items():
                        #     print(f"{k}: {v.item()}")
        print(f"before normalizing - tot_num_chunks_train - {tot_num_chunks_train}, tot_num_chunks_val - {tot_num_chunks_val}")
        train_loss = train_loss / tot_num_chunks_train
        val_loss = val_loss / tot_num_chunks_val
        train_loss_arr.append(train_loss)
        val_loss_arr.append(val_loss)
        print(f"{TimestampToString()}: Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        if True: #save checkpoint
            print(f"{TimestampToString()}: saving checkpoint on epoch {epoch + 1}")
            torch.save(model.state_dict(),f"RCNN_meas/Checkpoints/rcnn_weights_checkpoint_19_{int(epoch + 1)}.pth")
            np.savetxt("RCNN_meas/Stats/train_loss_arr_19.txt", train_loss_arr)
            np.savetxt("RCNN_meas/Stats/val_loss_arr_19.txt", val_loss_arr)