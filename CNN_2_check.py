import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from CNN_alt import CNNBoundingBoxModel
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNBoundingBoxModel().to(device)
model.load_state_dict(torch.load("Checkpoints3/cnn_weights_checkpoint_0_13.pth", map_location=device)) #9 | 15 | 18 | 19
model.eval()  # Set model to evaluation mode


h5_images = "Data/images.h5"
h5_masks = "Data/masks.h5"


with h5py.File(h5_images, 'r') as img5f:
    img = img5f['image_256'][:]
    image = np.transpose(img,(2, 0, 1))  # Shape: (128, 156, 128) - reorder so index 0 will be the layer (instead of 2)
    slice_data = np.expand_dims(image, axis=1)  # Shape: (128, 1, 128, 156)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

# # Display the first image
# axes[0].imshow(img[:,:,66], cmap='gray')
# axes[0].set_title('Image 1')
# axes[0].axis('off')  # Turn off axes
#
# # Display the second image
# axes[1].imshow(slice_data[66-55,0,:,:], cmap='gray')
# axes[1].set_title('Image 2')
# axes[1].axis('off')  # Turn off axes
#
# # Adjust layout and show the plot
# plt.tight_layout()
# plt.savefig('check_layers')

# Move input to the same device as the model

# Perform inference
chunk_size = 16
for chunk_idx in range(0,slice_data.shape[0],chunk_size):
    chunk = slice_data[chunk_idx:chunk_idx+chunk_size,:,:,:]
    print(chunk.shape)
    chunk_in = torch.tensor(chunk, dtype=torch.float32).to(device)
    with torch.no_grad():  # Disable gradient computation for faster inference
        predictions = model(chunk_in)  # Shape: (64, 4)
    # Convert predictions to CPU for further processing
    predictions = predictions.cpu().numpy()
    for i, pred in enumerate(predictions):
        print(f"Chunk {chunk_idx} - Slice {i}: Bounding Box: {pred}")
        if True:
            x_min, y_min, x_max, y_max, valid = np.round(pred).astype(int)
            plt.figure()
            plt.imshow(chunk[i,0,:,:], cmap='gray')
            plt.title(f"Chunk {chunk_idx} - Slice {i} with Predicted Box")
            plt.gca().add_patch(
                plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                              edgecolor='red', facecolor='none', lw=2)
            )
            # Show the plot
            plt.axis('off')  # Turn off axis labels
            plt.savefig(f"Results/Chunk_{chunk_idx}_Slice_{i}_Predicted_Box")





