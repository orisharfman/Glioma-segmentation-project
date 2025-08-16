import nibabel as nib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import matplotlib.pyplot as plt
import os
curr_dir = os.getcwd()
INPUT_PATH = "temp_input\BraTS-GLI-00000-000-t1c.nii.gz"
OUTPUT_PATH = "temp_results\preprocessed_slices"
# "C:\Users\yuval\PycharmProjects\SegmentationGUI\temp_input\BraTS-GLI-00000-000-t1c.nii.gz"
scaler = MinMaxScaler()

def preprocess_scan(InputPath , OutputPath = OUTPUT_PATH, plot=False):
    img_t1c = nib.load(InputPath).get_fdata()
    img_t1c = scaler.fit_transform(img_t1c.reshape(-1, img_t1c.shape[-1])).reshape(img_t1c.shape)
    img_t1c = img_t1c[56: 184, 44: 200, 13: 141]
    if plot:
        plot_slices(img_t1c)
    np.save(OutputPath, img_t1c)
    print("preprocess done")
    return img_t1c

def plot_slices(volume, indices=[128, 155, 127]):
    """
    Show specific slices from a volume in shape (D, 1, H, W).
    """
    for idx in range(indices[0]):
        slice_img = volume[:, :, idx]  # Remove channel dim
        plt.figure(figsize=(5, 5))
        plt.imshow(slice_img, cmap='gray')
        plt.title(f"Slice {idx}")
        plt.axis('off')
        plt.show()

# def preprocess_scan(input_path, show_slices=True, save_slices=True, output_dir="temp_results\preprocessed_slices"):
#     print("in function")
#     print(curr_dir)
#     # Load and crop
#     img = nib.load(input_path).get_fdata()
#     img = img[56:184, 44:200, 13:141]  # Crop → (128, 156, 128)
#
#     # Reorder to (D, H, W) then add channel: (D, 1, H, W)
#     img = np.transpose(img, (2, 0, 1))
#     img = np.expand_dims(img, axis=1)
#
#     # Normalize each slice
#     img_reshaped = img.reshape(128, -1)
#     scaler = MinMaxScaler(feature_range=(-1, 1))
#     img_normalized = scaler.fit_transform(img_reshaped)
#     img_normalized = img_normalized.reshape(128, 1, 128, 156)
#     plot_slices(img_normalized)
#     img_tensor = torch.tensor(img_normalized, dtype=torch.float32)
#
#     # Split into slices
#     image_list = list(img_tensor.unbind(0))  # 128 slices: each (1, 128, 156)
#
#     print(f"\nPreprocessing complete. Total slices: {len(image_list)}")
#
#     # Visualization
#     if show_slices or save_slices:
#         import os
#         if save_slices:
#             os.makedirs(output_dir, exist_ok=True)
#
#         for i, img_slice in enumerate(image_list):
#             print(f"Slice {i}: shape = {img_slice.shape}, min = {img_slice.min():.3f}, max = {img_slice.max():.3f}")
#             plt.figure(figsize=(4, 4))
#             plt.imshow(img_slice[0].numpy(), cmap='gray', vmin=-1, vmax=1)
#             plt.axis('off')
#             plt.title(f"Slice {i}")
#             if save_slices:
#                 plt.savefig(f"{output_dir}/slice_{i:03d}.png", bbox_inches='tight')
#             if show_slices:
#                 plt.show()
#             plt.close()
#
#     return image_list

# preprocess_scan(INPUT_PATH, OUTPUT_PATH)