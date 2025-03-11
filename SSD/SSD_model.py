import torch
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision

from skimage.transform import resize

def prepare_images(images):

    images = resize(images, (300, 300, 128), anti_aliasing=True, mode="reflect")
    images = np.transpose(images, (
        2, 0, 1))  # Shape: (128, 300, 300) - reorder so index 0 will be the layer (instead of 2)
    images = np.expand_dims(images, axis=1)  # Shape: (128, 1, 300, 300)
    images = np.repeat(images, 3, 1)
    images = torch.tensor(images, dtype=torch.float32)
    return images



# ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd')
ssd_model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
ssd_model.to(device)
ssd_model.eval()

h5_data_train = "/home/tauproj12/PycharmProjects/Glioma-segmentation-project/Data/train_data.h5"
h5_masks_train = "/home/tauproj12/PycharmProjects/Glioma-segmentation-project/Data/train_masks.h5"
h5_data_val = "/home/tauproj12/PycharmProjects/Glioma-segmentation-project/Data/val_data.h5"
h5_masks_val = "/home/tauproj12/PycharmProjects/Glioma-segmentation-project/Data/val_masks.h5"

h5_data_test = "/home/tauproj12/PycharmProjects/Glioma-segmentation-project/Data/test_data.h5"
h5_masks_test = "/home/tauproj12/PycharmProjects/Glioma-segmentation-project/Data/test_masks.h5"

train_loss_arr = []
val_loss_arr = []
num_epochs = 300

f_image = h5py.File(h5_data_train, "r")
image_keys = sorted(list(f_image.keys()))
print(image_keys[0])
image = f_image[image_keys[0]]
print(image)

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

    print(len(results_per_input))
    print(results_per_input[64])

    print(max(results_per_input[64][2]))




