import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from UNET_alt import UNet
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(n_class=4).to(device)
model.load_state_dict(torch.load("unet_weights_alt.pth", map_location=device))
model.eval()  # Set model to evaluation mode


h5_images = "Data/images.h5"
h5_masks = "Data/masks.h5"


with h5py.File(h5_images, 'r') as img5f:
    img = img5f['image_0'][:]
    image = np.transpose(img,(2, 0, 1))  # Shape: (128, 156, 128) - reorder so index 0 will be the layer (instead of 2)
    slice_data = np.expand_dims(image, axis=1)  # Shape: (128, 1, 128, 156)
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns
#
# # Display the first image
# axes[0].imshow(img[:,:,66], cmap='gray')
# axes[0].set_title('Image 1')
# axes[0].axis('off')  # Turn off axes
#
# # Display the second image
# axes[1].imshow(slice_data[66,0,:,:], cmap='gray')
# axes[1].set_title('Image 2')
# axes[1].axis('off')  # Turn off axes
#
# # Adjust layout and show the plot
# plt.tight_layout()
# plt.show()

# Move input to the same device as the model

# Perform inference
slice_data = torch.tensor(slice_data, dtype=torch.float32).to(device)
with torch.no_grad():  # Disable gradient computation for faster inference
    predictions = model(slice_data)  # Shape: (64, 4)

# Convert predictions to CPU for further processing
predictions = predictions.cpu().numpy()
for i, pred in enumerate(predictions):
    print(f"Slice {i}: Bounding Box: {pred}")
    if i == 43:
        x_min, y_min, x_max, y_max, valid = np.round(pred).astype(int)
        plt.imshow(img[:,:,66], cmap='gray')
        plt.title(f"Slice {i} with Predicted Box")
        plt.gca().add_patch(
            plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                          edgecolor='red', facecolor='none', lw=2)
        )
        # Show the plot
        plt.axis('off')  # Turn off axis labels
        plt.show()





