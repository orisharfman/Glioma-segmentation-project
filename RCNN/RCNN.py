import torchvision
from torchvision import  transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch import no_grad
from google.protobuf.internal.well_known_types import Timestamp
from GeneralFunctions import TimestampToString


import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from colorama import Fore
from colorama import Style


def prepare_images(images):
    images = np.transpose(images, (
        2, 0, 1))  # Shape: (128, 156, 128) - reorder so index 0 will be the layer (instead of 2)
    images = np.expand_dims(images, axis=1)  # Shape: (128, 1, 128, 156)
    images = torch.tensor(images, dtype=torch.float32)
    images = list(images.unbind(0))
    return images

def prepare_targrets(masks):
    targets = []  # input as list of dictionaries

    for z in range(masks.shape[2]):  # for each slice
        slice_2d = masks[:, :, z]
        rows, cols = np.where(slice_2d > 0)
        if rows.size > 0 and cols.size > 0:
            x_min, x_max = cols.min() - 3, cols.max() + 3
            y_min, y_max = rows.min() - 3, rows.max() + 3
            targets.append({
                "boxes": torch.tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.int64),
                "image_id": torch.tensor([z]),
                "area": torch.tensor([(x_max - x_min) * (y_max - y_min)], dtype=torch.float32),
                "iscrowd": torch.empty((0,), dtype=torch.int64),
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


    num_classes = 2  # only 1 class to find, and additional for background.
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    print(model)
    exit(0)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)


    # checkpoint = torch.load("RCNN_meas/Checkpoints/rcnn_weights_checkpoint_0_2.pth",
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

    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    model.to(device)




    h5_data_train = "Data/train_data.h5"
    h5_masks_train = "Data/train_masks.h5"

    h5_data_val = "Data/val_data.h5"
    h5_masks_val = "Data/val_masks.h5"

    h5_data_test = "Data/test_data.h5"
    h5_masks_test = "Data/test_masks.h5"

    train_loss_arr = []
    val_loss_arr = []
    num_epochs = 300

    with h5py.File(h5_data_train, "r") as f_image, h5py.File(h5_masks_train, "r") as f_mask:
        image_keys = sorted(list(f_image.keys()))
        mask_keys = sorted(list(f_mask.keys()))
        train_size = len(image_keys)
    with h5py.File(h5_data_val, "r") as f_image, h5py.File(h5_masks_val, "r") as f_mask:
        image_val_keys = sorted(list(f_image.keys()))
        mask_val_keys = sorted(list(f_mask.keys()))
        val_size = len(image_val_keys)
    print(f"{TimestampToString()}: start training, train_size = {train_size}, val_size = {val_size}")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        with h5py.File(h5_data_train, "r") as f_image, h5py.File(h5_masks_train, "r") as f_mask:
            for key_idx in range(len(image_keys)):
                if mask_keys[key_idx].split("_")[1]!=image_keys[key_idx].split("_")[1]: #sanity for keys
                    print(f"not matching keys! {mask_keys[key_idx]}~{image_keys[key_idx]}")
                    exit(-1)
                images = f_image[image_keys[key_idx]]
                images = prepare_images(images)

                masks = f_mask[mask_keys[key_idx]] # Shape: (128, 156, 128, 2)
                masks = masks[:, :, :, 1].astype(np.uint8)  # Shape: (128, 156, 128)
                targets = prepare_targrets(masks)
                chunk_size = 8
                for i in range(0,len(images),chunk_size):
                    images_chunk = images[i:i+chunk_size]
                    targets_chunk = targets[i:i+chunk_size]

                    images_chunk = [img.to(device) for img in images_chunk]
                    targets_chunk = [{k: v.to(device) for k, v in t.items()} for t in targets_chunk]
                    loss_dict = model(images_chunk, targets_chunk)
                    loss = sum(loss for loss in loss_dict.values())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

        print(f"{TimestampToString()}: validation")
        val_loss = 0

        with h5py.File(h5_data_val, "r") as f_image, h5py.File(h5_masks_val, "r") as f_mask:
            with torch.no_grad():
                for key_idx in range(len(image_val_keys)):
                    if mask_val_keys[key_idx].split("_")[1] != image_val_keys[key_idx].split("_")[1]:  # sanity for keys
                        print(f"not matching keys! {mask_keys[key_idx]}~{image_val_keys[key_idx]}")
                        exit(-1)
                    images = f_image[image_val_keys[key_idx]]
                    images = prepare_images(images)

                    masks = f_mask[mask_val_keys[key_idx]]  # Shape: (128, 156, 128, 2)
                    masks = masks[:, :, :, 1].astype(np.uint8)  # Shape: (128, 156, 128)
                    targets = prepare_targrets(masks)

                    chunk_size = 8
                    for i in range(0, len(images), chunk_size):
                        images_chunk = images[i:i + chunk_size]
                        targets_chunk = targets[i:i + chunk_size]
                        images_chunk = [img.to(device) for img in images_chunk]
                        targets_chunk = [{k: v.to(device) for k, v in t.items()} for t in targets_chunk]
                        # Run the model on the validation images
                        loss_dict = model(images_chunk, targets_chunk)  # Get the loss dictionary
                        loss = sum(loss for loss in loss_dict.values())
                        val_loss += loss.item()# Total loss for this batch

        train_loss = train_loss / (train_size*16)
        val_loss = val_loss / (val_size*16)
        train_loss_arr.append(train_loss)
        val_loss_arr.append(val_loss)
        print(f"{TimestampToString()}: Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        if True: #save checkpoint
            print(f"{TimestampToString()}: saving checkpoint on epoch {epoch + 1}")
            torch.save(model.state_dict(),f"RCNN_meas/Checkpoints/rcnn_weights_checkpoint_2_{int(epoch + 1)}.pth")
            np.savetxt("RCNN_meas/Stats/train_loss_arr_2.txt", train_loss_arr)
            np.savetxt("RCNN_meas/Stats/val_loss_arr_2.txt", val_loss_arr)