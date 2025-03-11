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

    model.eval()  # Set model to evaluation mode
    h5_images = "Data/images.h5"
    h5_masks = "Data/masks.h5"
    color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    with h5py.File(h5_masks, 'r') as mask5f:
        print(mask5f.keys())
        mask = mask5f['mask_106'][:]
        mask = mask[:,:,:,1]
    with h5py.File(h5_images, 'r') as img5f:
        img = img5f['image_106'][:]
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
        for c in range(0, len(images), chunk_size):
            images_chunk = images[c:c+chunk_size]
            images_chunk_ = [img.to(device) for img in images_chunk]
            predictions = model(images_chunk_)
            print(predictions)
            bbox = predictions[0]['boxes'].cpu().detach().numpy()
            mask_chunk = mask[:,:,c:c+chunk_size]
            for i in range(len(images_chunk)):
                plt.figure()
                plt.imshow(images_chunk[i][0, :, :], cmap='gray')
                mask_ci = mask_chunk[:, :,i]
                h, w = mask_ci.shape[-2:]
                plt.imshow(mask_ci.reshape(h, w, 1) * color.reshape(1, 1, -1))
                plt.title(f"Example of box prediction")

                for p in range(len(predictions[0]['labels'])):
                    if predictions[0]['labels'][p] == 1:
                        if predictions[0]['scores'][p]<0.7:
                            continue
                        x_min, y_min, x_max, y_max =  np.round(bbox[i]).astype(int)
                        plt.gca().add_patch(
                            plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                          edgecolor='red', facecolor='none', lw=2)
                        )
                plt.axis('off')
                plt.show()

