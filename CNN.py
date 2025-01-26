import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from google.protobuf.internal.well_known_types import Timestamp
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from GeneralFunctions import TimestampToString
from OurLossFunctions import MaskedCombinedLoss

class CNNBoundingBoxModel(nn.Module):
    def __init__(self):
        super(CNNBoundingBoxModel, self).__init__()
        # First block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # Shrink image size
        self.dropout1 = nn.Dropout(p=0.25)  # Dropout with 25% probability

        # Second block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # Shrink image size
        self.dropout2 = nn.Dropout(p=0.25)  # Dropout with 25% probability

        # Third block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # Shrink image size
        self.dropout3 = nn.Dropout(p=0.25)  # Dropout with 25% probability

        # Fourth block (optional, deeper network)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)  # Shrink image size
        self.dropout4 = nn.Dropout(p=0.25)  # Dropout with 25% probability

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 8 * 9, 512)  # Adjust based on final image size
        self.dropout_fc1 = nn.Dropout(p=0.5)  # Dropout with 50% probability
        self.fc2 = nn.Linear(512, 5)  # Output 4 values: x_min, y_min, x_max, y_max


    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.dropout1(x)  # Apply dropout after first block

        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.dropout2(x)  # Apply dropout after second block

        x = self.pool3(torch.relu(self.conv3(x)))
        x = self.dropout3(x)  # Apply dropout after third block

        x = self.pool4(torch.relu(self.conv4(x)))  # Optional deeper layer
        x = self.dropout4(x)  # Apply dropout after forth block

        x = x.view(-1, 256 * 8 * 9)  # Flatten for fully connected layer
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
    model.load_state_dict(torch.load("Checkpoints/cnn_weights_checkpoint_1_9.pth", map_location=device))
    criterion = MaskedCombinedLoss(1.0,0.8)   # Loss of the output box and the Objectness index (L1 + BCEWithLogitsLoss)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

    scaler = torch.amp.GradScaler('cuda')  # Initialize GradScaler for mixed precision
    epochs = 100
    chunk_size = 16

    print(f"{TimestampToString()}: start training")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        # if epoch < 5:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = 1e-3
        # elif epoch < 30:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = 1e-4
        # else:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = 1e-4

        for slices, labels in train_loader:
            for i in range(0,slices.shape[0],chunk_size):
                slices_i = slices[i:i+chunk_size,:,:,:] # slices_i: (16, 1, 128, 156)
                labels_i = labels[i:i + chunk_size, :]  # slices_i: (16, 5)
                # Move to device
                slices_i, labels_i = slices_i.to(device), labels_i.to(device)  # slices: (1, 16, 1, 128, 156), labels: (1, 16, 5)

                # Flatten the batch of slices
                slices_i = slices_i.view(-1, 1, 128, 156)  # Shape: (16, 1, 128, 156)
                labels_i = labels_i.view(-1, 5)  # Shape: (16, 5)


                # Forward pass
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    outputs = model(slices_i)  # Shape: (16, 5)
                    loss = criterion(outputs, labels_i)

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
        with torch.no_grad():
            for slices, labels in val_loader:
                slices, labels = slices.to(device), labels.to(device)
                slices = slices.view(-1, 1, 128, 156)
                labels = labels.view(-1, 5)

                outputs = model(slices)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * slices.size(0)

        val_loss /= len(val_loader.dataset) * 128

        print(f"{TimestampToString()}: Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        if (epoch+1)%10 == 0 and epoch >0:
            print(f"saving checkpoint on epoch {epoch+1}")
            torch.save(model.state_dict(), f"Checkpoints2/cnn_weights_checkpoint_2_{int((epoch+1)/10)}.pth")
            print("saved")