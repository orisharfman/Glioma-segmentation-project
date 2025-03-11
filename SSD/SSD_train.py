import torch
import torchvision.transforms as transforms

import torch.nn as nn
import torch.optim as optim

import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from skimage.transform import resize
from google.protobuf.internal.well_known_types import Timestamp
from GeneralFunctions import TimestampToString

def prepare_images(images):

    images = resize(images, (300, 300, 128), anti_aliasing=True, mode="reflect")
    images = np.transpose(images, (
        2, 0, 1))  # Shape: (128, 300, 300) - reorder so index 0 will be the layer (instead of 2)
    images = np.expand_dims(images, axis=1)  # Shape: (128, 1, 300, 300)
    images = np.repeat(images, 3, 1)
    images = torch.tensor(images, dtype=torch.float32)
    return images

def prepare_targrets(masks):
    masks = resize(masks, (300, 300, 128), anti_aliasing=True, mode="reflect")

    targets = []  # input as list of dictionaries
    boxes = []
    labels = []
    for z in range(masks.shape[2]):  # for each slice
        slice_2d = masks[:, :, z]
        rows, cols = np.where(slice_2d > 0)
        if rows.size > 0 and cols.size > 0:
            x_min, x_max = cols.min() - 3, cols.max() + 3
            y_min, y_max = rows.min() - 3, rows.max() + 3
            boxes.append(torch.tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float32)/300)
            labels.append(torch.tensor([1], dtype=torch.int64))
        else:
            # If the slice has no non-zero elements, add empty list
            boxes.append(torch.zeros((0, 4), dtype=torch.float32))
            labels.append(torch.empty((0,), dtype=torch.int64))
    return [boxes, labels]

if __name__ == "__main__":


    ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd')
    utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')

    # criterion = nn.CrossEntropyLoss()  # For classification
    # regression_loss = nn.SmoothL1Loss()  # For bounding boxes
    optimizer = optim.Adam(ssd_model.parameters(), lr=0.00001)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ssd_model.to(device)



    h5_data_train = "/home/tauproj12/PycharmProjects/Glioma-segmentation-project/Data/train_data.h5"
    h5_masks_train = "/home/tauproj12/PycharmProjects/Glioma-segmentation-project/Data/train_masks.h5"
    h5_data_val = "/home/tauproj12/PycharmProjects/Glioma-segmentation-project/Data/val_data.h5"
    h5_masks_val = "/home/tauproj12/PycharmProjects/Glioma-segmentation-project/Data/val_masks.h5"
    h5_data_test = "/home/tauproj12/PycharmProjects/Glioma-segmentation-project/Data/test_data.h5"
    h5_masks_test = "/home/tauproj12/PycharmProjects/Glioma-segmentation-project/Data/test_masks.h5"

    train_loss_arr = []
    val_loss_arr = []
    num_epochs = 300

    f_image_train = h5py.File(h5_data_train, "r")
    f_masks_train = h5py.File(h5_masks_train, "r")
    f_image_val = h5py.File(h5_data_val, "r")
    f_masks_val = h5py.File(h5_masks_val, "r")
    image_train_keys = sorted(list(f_image_train.keys()))
    masks_train_keys = sorted(list(f_masks_train.keys()))
    image_val_keys = sorted(list(f_image_val.keys()))
    masks_val_keys = sorted(list(f_masks_val.keys()))
    train_size = len(image_train_keys)
    val_size = len(image_val_keys)
    print(f"{TimestampToString()}: start training, train_size = {train_size}, val_size = {val_size}")
    for epoch in range(num_epochs):
        ssd_model.train()
        train_loss = 0
        for key_idx in range(len(image_train_keys)):
            if image_train_keys[key_idx].split("_")[1] != image_val_keys[key_idx].split("_")[1]:  # sanity for keys
                print(f"not matching keys! {image_train_keys[key_idx]}~{image_val_keys[key_idx]}")
                exit(-1)
            images = f_image_train[image_train_keys[key_idx]]
            images = prepare_images(images)
            masks = f_masks_train[masks_train_keys[key_idx]]  # Shape: (128, 156, 128, 2)
            masks = masks[:, :, :, 1].astype(np.uint8)  # Shape: (128, 156, 128)
            boxes,labels = prepare_targrets(masks)
            optimizer.zero_grad()
            detections_batch  = ssd_model(images)  # Outputs: (128, 4, 8732), (128, 81, 8732)
            bbox_preds, class_preds  = detections_batch
            results_per_input = utils.decode_results(detections_batch)
            best_results_per_input = [utils.pick_best(results, 0.4) for results in results_per_input]
            class_loss = criterion(class_preds, labels)  # Classification loss
            bbox_loss = regression_loss(bbox_preds, boxes)  # Localization loss
            loss = ssd_loss(bbox_preds, class_preds, boxes, labels)
            loss = class_loss + bbox_loss
            train_loss
            loss.backward()
            optimizer.step()

images = prepare_images(image)
images = images.to(device)
with torch.no_grad():
    detections_batch = ssd_model(images)
    results_per_input = utils.decode_results(detections_batch)

    print(len(detections_batch))
    print(detections_batch[0].shape)
    print(detections_batch[1].shape)
    print(detections_batch[1])
    print("-----------------")

    best_results_per_input = [utils.pick_best(results, 0.4) for results in results_per_input]

    print(best_results_per_input)




