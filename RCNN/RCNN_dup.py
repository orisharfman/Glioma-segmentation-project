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

class DynamicKeySliceDataset(Dataset):
    def __init__(self,h5_images, h5_masks, transform=None):
        self.transform = transform
        self.h5_images = h5_images
        self.h5_masks = h5_masks
        # Open the HDF5 files to get the list of keys
        with h5py.File(h5_images, 'r') as f:
            self.image_keys = [key for key in f.keys()]
        with h5py.File(h5_masks, 'r') as f:
            self.masks_keys = [key for key in f.keys()]

        # Sort keys to ensure consistent ordering
        self.image_keys.sort()
        self.masks_keys.sort()

    def __len__(self):
        return len(self.image_keys)

    def __getitem__(self, idx):
        # Open the HDF5 file for reading
        with h5py.File(self.h5_images, 'r') as f:
            # print(f"processing {self.image_keys[idx]}")
            image = f[self.image_keys[idx]][:]  # Shape: (128, 156, 128)
            image = np.transpose(image, (
            2, 0, 1))  # Shape: (128, 156, 128) - reorder so index 0 will be the layer (instead of 2)
        with h5py.File(self.h5_masks, 'r') as f:
            mask = f[self.masks_keys[idx]][:]  # Shape: (128, 156, 128, 2)
            mask = mask[:, :, :, 1].astype(np.uint8)  # Shape: (128, 156, 128)
            bounding_boxes = []
            targets = []
            for z in range(mask.shape[2]):
                slice_2d = mask[:, :, z]
                rows, cols = np.where(slice_2d > 0)
                if rows.size > 0 and cols.size > 0:
                    x_min, x_max = cols.min(), cols.max()
                    y_min, y_max = rows.min(), rows.max()
                    targets.append({
                        "boxes": torch.tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float32),
                        "labels": torch.tensor([1], dtype=torch.int64),
                        "image_id": torch.tensor([z]),
                        "area": torch.tensor([(x_max-x_min)*(y_max-y_min)], dtype=torch.float32),
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
                    if targets[-1]['boxes'].dim() == 3:
                        targets[-1]['boxes'] = targets[-1]['boxes'].squeeze(0).view(-1, 4)
        # Convert to tensors
        # image = np.expand_dims(image, axis=1)  # Shape: (128, 1, 128, 156)
        # image = torch.tensor(image, dtype=torch.float32)
        # image = list(image.unbind(0))
        return image, mask


if __name__ == "__main__":

    torch.cuda.empty_cache()  # Clears unused memory from cache
    torch.cuda.ipc_collect()  # Forces garbage collection of unused tensors


    num_classes = 2  # only 1 class to find, and additional for background.
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)


    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    checkpoint = torch.load("RCNN_meas/Checkpoints/rcnn_weights_checkpoint_0_2.pth",
                            map_location=device)  # Change to 'cuda' if using GPU
    model.load_state_dict(checkpoint)

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





    # Load dataset
    h5_images = "Data/images.h5"
    h5_masks = "Data/masks.h5"
    dataset = DynamicKeySliceDataset(h5_images, h5_masks)
    train_size = int(0.75 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = int(0.05 * len(dataset))
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    batch_size = 1  # One 3D image (128 slices) per batch

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation data
        num_workers=4,  # Same as training for parallel loading
        pin_memory=True,  # Useful for GPU acceleration
        prefetch_factor=2  # Prefetch batches for efficiency
    )


    train_loss_arr = []
    val_loss_arr = []
    num_epochs = 300

    print(f"{TimestampToString()}: start training, train_size = {train_size}, val_size = {val_size}")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, masks in train_loader:
            # Convert to list
            images = np.expand_dims(images[0,:,:,:], axis=1)  # Shape: (128, 1, 128, 156)
            images = torch.tensor(images, dtype=torch.float32)
            images = list(images.unbind(0))
            masks = masks[0,:,:,:]
            targets = []#input as list of dictionaries

            for z in range(masks.shape[2]):#for each slice
                slice_2d = masks[:, :, z]
                rows, cols = np.where(slice_2d > 0)
                if rows.size > 0 and cols.size > 0:
                    x_min, x_max = cols.min()-3, cols.max()+3
                    y_min, y_max = rows.min()-3, rows.max()+3
                    targets.append({
                        "boxes": torch.tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float32),
                        "labels": torch.tensor([1], dtype=torch.int64),
                        "image_id": torch.tensor([z]),
                        "area": torch.tensor([(x_max-x_min)*(y_max-y_min)], dtype=torch.float32),
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
            # Move to device
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
        # all_preds = []
        # all_targets = []
        with torch.no_grad():
            for images, masks in val_loader:
                images = np.expand_dims(images[0, :, :, :], axis=1)  # Shape: (128, 1, 128, 156)
                images = torch.tensor(images, dtype=torch.float32)
                images = list(images.unbind(0))
                masks = masks[0, :, :, :]
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
                # Move to device
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
                    # predictions = model(images) # Store predictions and targets for evaluation
        train_loss = train_loss / train_size
        val_loss = val_loss / val_size
        train_loss_arr.append(train_loss)
        val_loss_arr.append(val_loss)
        print(f"{TimestampToString()}: Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        if (epoch + 1) % 3 == 0 and epoch > 0: #save checkpoint
            print(f"{TimestampToString()}: saving checkpoint on epoch {epoch + 1}")
            torch.save(model.state_dict(),f"RCNN_meas/Checkpoints/rcnn_weights_checkpoint_1_{int(epoch + 1)}.pth")
            np.savetxt("RCNN_meas/Stats/train_loss_arr_cont.txt", train_loss_arr)
            np.savetxt("RCNN_meas/Stats/val_loss_arr_cont.txt", val_loss_arr)

    # for slices, labels in train_loader:
    #     slices = slices[0,66,:,:,:]
    #     print(f"f = {slices.shape}")
    #     pred = model_([slices])
    #     scores = pred[0]['scores'].detach().numpy()
    #     boxes = pred[0]['boxes'].detach()
    #     plt.figure()
    #     plt.imshow(slices[0, :, :], cmap='gray')
    #     for i in range(len(scores)):
    #         if scores[i]>0.5:
    #             print(f"score is {scores[i]}")
    #             print(boxes[i])
    #             x_min, y_min, x_max, y_max = boxes[i]
    #             plt.gca().add_patch(
    #                 plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
    #                               edgecolor='red', facecolor='none', lw=2)
    #             )
    #     plt.show()
    #     exit(0)