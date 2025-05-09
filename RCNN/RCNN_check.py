import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torchvision
from torchvision import  transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch import no_grad
from google.protobuf.internal.well_known_types import Timestamp
from GeneralFunctions import TimestampToString, create_plots


import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from colorama import Fore
from colorama import Style
from sklearn.preprocessing import MinMaxScaler




def prepare_images(images):
    images = np.transpose(images, (
        2, 0, 1))  # Shape: (128, 156, 128) - reorder so index 0 will be the layer (instead of 2)
    images = np.expand_dims(images, axis=1)  # Shape: (128, 1, 128, 156)

    # Reshape images to 2D (flatten each image) for MinMaxScaler
    images_reshaped = images.reshape(128, -1)  # Shape: (128, 19968)

    # Apply MinMaxScaler with feature range (-1,1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    normalized_images = scaler.fit_transform(images_reshaped)

    # Reshape back to original shape (128, 1, 128, 156)
    images = normalized_images.reshape(128, 1, 128, 156)

    images = torch.tensor(images, dtype=torch.float32)
    images = list(images.unbind(0))
    return images

if __name__ == "__main__":

    torch.cuda.empty_cache()  # Clears unused memory from cache
    torch.cuda.ipc_collect()  # Forces garbage collection of unused tensors


    num_classes = 2  # only 1 class to find, and additional for background.
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)


    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    checkpoint = torch.load("RCNN_meas/Checkpoints/rcnn_weights_checkpoint_18_15.pth",
                            map_location=device)  # Change to 'cuda' if using GPU
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()  # Set model to evaluation mode
    h5_images = "/home/eprojuser008/GliomaSegmentation/Glioma-segmentation-project/Data/val_data.h5"
    h5_masks = "/home/eprojuser008/GliomaSegmentation/Glioma-segmentation-project/Data/val_masks.h5"
    train_loss_path = "RCNN_meas/Stats/train_loss_arr_19.txt"
    val_loss_path = "RCNN_meas/Stats/val_loss_arr_19.txt"
    loss_fig_path = "RCNN_meas/Stats/loss_fig_19.png"

    color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    create_plots(train_loss_path, val_loss_path, loss_fig_path)
    exit()
    with h5py.File(h5_masks, 'r') as mask5f:
        print(mask5f.keys())
        mask = mask5f['mask_13'][:]
        mask = mask[:,:,:,1] + mask[:,:,:,2]
    with h5py.File(h5_images, 'r') as img5f:
        img = img5f['image_13'][:]
        images = prepare_images(img)
        image = np.transpose(img,(2, 0, 1))  # Shape: (128, 156, 128) - reorder so index 0 will be the layer (instead of 2)
        image = np.expand_dims(image, axis=1)  # Shape: (128, 1, 128, 156)
        image = torch.tensor(image, dtype=torch.float32)
        images = list(image.unbind(0))
        images_chunk = images[66:70]
        images_chunk_ = [img.to(device) for img in images_chunk]
        # predictions = model(images_chunk_)
        # bbox = predictions[0]['boxes'].cpu().detach().numpy()
        # print(bbox)
        chunk_size = 1
        for c in range(1, len(images), chunk_size):
            images_chunk = images[c:c+chunk_size]
            images_chunk_ = [img.to(device) for img in images_chunk]
            predictions = model(images_chunk_)
            for pred in predictions:
                print(f"Boxes: {pred['boxes']}")
                print(f"Scores: {pred['scores']}")
            #print(predictions)
            bbox = predictions[0]['boxes'].cpu().detach().numpy()
            mask_chunk = mask[:,:,c:c+chunk_size]
            for i in range(len(images_chunk)):
                plt.figure()
                plt.imshow(images_chunk[i][0, :, :], cmap='gray')
                mask_ci = mask_chunk[:, :,i]
                h, w = mask_ci.shape[-2:]
                # plt.imshow(mask_ci.reshape(h, w, 1) * color.reshape(1, 1, -1))
                plt.title(f"Example of box prediction")

                for p in range(len(predictions[0]['labels'])):
                    if predictions[0]['labels'][p] == 1:
                        if predictions[0]['scores'][p]<0.75:
                            continue
                        x_min, y_min, x_max, y_max =  np.round(bbox[p]).astype(int)
                        plt.gca().add_patch(
                            plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                          edgecolor='red', facecolor='none', lw=2)
                        )
                plt.axis('off')
                plt.savefig(f"RCNN_meas/Check/18_15/RCNN_check_no_mask_img_13_{c}", bbox_inches='tight')
                print(f"saved to RCNN_meas/Check/18_15/RCNN_check_no_mask_img_13_{c}")
                plt.close()

