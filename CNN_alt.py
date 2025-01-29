import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from google.protobuf.internal.well_known_types import Timestamp
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from GeneralFunctions import TimestampToString, accuracy_calc, create_plots
from OurLossFunctions import MaskedCombinedLoss, GiouCombinedLoss


class CNNBoundingBoxModel(nn.Module):
    def __init__(self):
        super(CNNBoundingBoxModel, self).__init__()
        # First block
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2) # output: 128x156x64
        self.pool1 = nn.MaxPool2d(2, 2)  # Shrink image size                    # output: 64x78x64
        self.dropout1 = nn.Dropout(p=0.25)  # Dropout with 25% probability

        # Second block
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2) # output: 64x78x128
        self.pool2 = nn.MaxPool2d(2, 2)  # Shrink image size                     # output: 32x39x128
        self.dropout2 = nn.Dropout(p=0.25)  # Dropout with 25% probability

        # Third block
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)  # output: 32x39x256
        self.pool3 = nn.MaxPool2d(2, 2)  # Shrink image size                       # output: 16x19x256
        self.dropout3 = nn.Dropout(p=0.25)  # Dropout with 25% probability

        # Fourth block (optional, deeper network)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)  # output: 16x19x512
        self.pool4 = nn.MaxPool2d(2, 2)  # Shrink image size                       # output: 8x9x512
        self.dropout4 = nn.Dropout(p=0.25)  # Dropout with 25% probability

        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)  # output: 8x9x512

        # Fully connected layers
        self.fc1 = nn.Linear(1024 * 8 * 9, 1024)  # Adjust based on final image size
        self.dropout_fc1 = nn.Dropout(p=0.5)  # Dropout with 50% probability
        self.fc2 = nn.Linear(1024, 5)  # Output 5 values: x_min, y_min, x_max, y_max, objectness


    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.dropout1(x)  # Apply dropout after first block

        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.dropout2(x)  # Apply dropout after second block

        x = self.pool3(torch.relu(self.conv3(x)))
        x = self.dropout3(x)  # Apply dropout after third block

        x = self.pool4(torch.relu(self.conv4(x)))  # Optional deeper layer
        x = self.dropout4(x)  # Apply dropout after forth block

        x = torch.relu(self.conv5(x))

        x = x.view(-1, 1024 * 8 * 9)  # Flatten for fully connected layer
        x = torch.relu(self.fc1(x))
        x = self.dropout_fc1(x)  # Apply dropout after first FC layer

        x = self.fc2(x)  # Output bounding box coordinates
        return x


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
            for z in range(mask.shape[2]):
                slice_2d = mask[:, :, z]
                rows, cols = np.where(slice_2d > 0)
                if rows.size > 0 and cols.size > 0:
                    x_min, x_max = cols.min(), cols.max()
                    y_min, y_max = rows.min(), rows.max()
                    bounding_boxes.append([x_min, y_min, x_max, y_max, 1])  # last index is objectness indicator
                else:
                    # If the slice has no non-zero elements, add a default box ((0, 0, 0, 0 , 0))
                    bounding_boxes.append([0, 0, 0, 0, 0])  # no object to identify
            mask = np.array(bounding_boxes)  # Shape: (128,5)
        # Normalize image slices and add channel dimension
        slices = np.expand_dims(image, axis=1)  # Shape: (128, 1, 128, 156)

        # Convert to tensors
        slices = torch.tensor(slices, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        return slices, mask

if __name__ == "__main__":
    # Load dataset
    h5_images = "Data/images.h5"
    h5_masks = "Data/masks.h5"

    dataset = DynamicKeySliceDataset(h5_images,h5_masks)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    print(f"start training, train_size = {train_size}, val_size = {val_size}")
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # DataLoader
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


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNBoundingBoxModel().to(device)
    model.load_state_dict(torch.load("Checkpoints6/cnn_weights_checkpoint_0_20.pth", map_location=device))
    criterion = MaskedCombinedLoss(1,0.8)   # Loss of the output box and the Objectness index (L1 + BCEWithLogitsLoss)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

    scaler = torch.amp.GradScaler('cuda')  # Initialize GradScaler for mixed precision
    epochs = 200
    chunk_size = 128

    print(f"{TimestampToString()}: start training")

    val_loss_arr = []
    train_loss_arr = []
    train_confusion_arr = []
    train_iou_arr = []
    validate_iou_arr = []
    validate_confusion_arr = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        if epoch < -1:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-4
        elif epoch < -1:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 5e-5
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 3e-6
        train_accuracy = [0, 0, 0, 0, 0]
        for slices, labels in train_loader:
            for i in range(0,slices.shape[0],chunk_size):
                slices_i = slices[:,i:i+chunk_size,:,:,:] # slices_i: (16, 1, 128, 156)
                labels_i = labels[:,i:i + chunk_size, :]  # slices_i: (16, 5)

                # Move to device
                slices_i, labels_i = slices_i.to(device), labels_i.to(device)  # slices: (1, 128, 1, 128, 156), labels: (1, 128, 5)

                # Flatten the batch of slices
                slices_i = slices_i.view(-1, 1, 128, 156)  # Shape: (128, 1, 128, 156)
                labels_i = labels_i.view(-1, 5)  # Shape: (128, 5)


                # Forward pass
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    outputs = model(slices_i)  # Shape: (128, 5)
                    loss = criterion(outputs, labels_i)
                    train_accuracy_value = accuracy_calc(outputs, labels_i)
                    train_accuracy = np.add(train_accuracy,train_accuracy_value)
                # Backward pass
                scaler.scale(loss).backward()

                # Gradient clipping
                max_norm = 1.0  # Maximum norm for gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item() * slices_i.size(0)

        train_loss /= len(train_loader.dataset) * 128  # Normalize by total slices (not really accurate, but it works fine so I won't interrupt it)

        # Validation loop
        model.eval()
        val_loss = 0.0
        validate_accuracy = [0, 0, 0, 0, 0]
        with torch.no_grad():
            for slices, labels in val_loader:
                slices, labels = slices.to(device), labels.to(device)
                slices = slices.view(-1, 1, 128, 156)
                labels = labels.view(-1, 5)

                outputs = model(slices)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * slices.size(0)
                validate_accuracy_value = accuracy_calc(outputs, labels)
                validate_accuracy = np.add(validate_accuracy,validate_accuracy_value)

        val_loss /= len(val_loader.dataset) * 128
        train_loss_arr.append(train_loss)
        val_loss_arr.append(val_loss)
        print(f"{TimestampToString()}: Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if train_accuracy[0] > 0:
            train_iou = train_accuracy[4]/train_accuracy[0] # iou_sum / true_positive_cnt
        else:
            train_iou = 0
        train_iou_arr.append(train_iou)

        train_tot = np.sum(train_accuracy[:4])
        train_confusion = train_accuracy[:4]/train_tot
        train_confusion_matrix = np.reshape(train_confusion,(2,2))
        train_confusion_arr.append(train_confusion)
        print(f"Train Stats: iou: {train_iou}, confusionmatrix:")
        print(train_confusion_matrix)

        if validate_accuracy[0] > 0:
            validate_iou = validate_accuracy[4] / validate_accuracy[0]  # iou_sum / true_positive_cnt
        else:
            validate_iou = 0
        validate_iou_arr.append(validate_iou)

        validate_tot = np.sum(validate_accuracy[:4])
        validate_confusion = validate_accuracy[:4] / validate_tot
        validate_confusion_matrix = np.reshape(validate_confusion, (2, 2))
        validate_confusion_arr.append(validate_confusion)
        print(f"Validation Stats: iou: {validate_iou}, confusionmatrix:")
        print(validate_confusion_matrix)

        if (epoch+1)%10 == 0 and epoch >0:
            print(f"saving checkpoint on epoch {epoch+1}")
            torch.save(model.state_dict(), f"Checkpoints6/cnn_weights_checkpoint_1_{int((epoch+1)/10)}.pth")
            print("saved")
            np.savetxt("Stats3/train_loss_arr.txt", train_loss_arr)
            np.savetxt("Stats3/val_loss_arr.txt", val_loss_arr)
            np.savetxt("Stats3/train_confusion_arr.txt", train_confusion_arr)
            np.savetxt("Stats3/validate_confusion_arr.txt", validate_confusion_arr)
            np.savetxt("Stats3/train_iou_arr.txt", train_iou_arr)
            np.savetxt("Stats3/validate_iou_arr.txt", validate_iou_arr)

    path_to_train_loss = "Stats2/train_loss_arr.txt"
    path_to_validate_loss = "Stats2/val_loss_arr.txt"
    path_to_train_iou = "Stats2/train_iou_arr.txt"
    path_to_validate_iou = "Stats2/validate_iou_arr.txt"
    create_plots(path_to_train_loss, path_to_validate_loss, path_to_train_iou, path_to_validate_iou)
