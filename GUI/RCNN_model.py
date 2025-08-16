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
# from google.protobuf.internal.well_known_types import Timestamp
# from GeneralFunctions import TimestampToString, create_plots
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# from colorama import Fore
# from colorama import Style
from sklearn.preprocessing import MinMaxScaler
from skimage.measure import label, regionprops

CHECKPOINT_PATH = os.path.join("checkpoints", "rcnn_weights_checkpoint_19_30.pth")
PREPROCESSED_IMAGE_PATH = "temp_results\preprocessed_slices.npy"

def prepare_images(images):
    images = np.transpose(images, (2, 0, 1))  # Shape: (128, 156, 128) - reorder so index 0 will be the layer (instead of 2)
    images = np.expand_dims(images, axis=1)  # Shape: (128, 1, 128, 156)

    # Reshape images to 2D (flatten each image) for MinMaxScaler
    # images_reshaped = images.reshape(128, -1)  # Shape: (128, 19968)

    # # Apply MinMaxScaler with feature range (-1,1)
    # scaler = MinMaxScaler(feature_range=(-1, 1))
    # normalized_images = scaler.fit_transform(images_reshaped)
    #
    # # Reshape back to original shape (128, 1, 128, 156)
    # images = normalized_images.reshape(128, 1, 128, 156)

    images = torch.tensor(images, dtype=torch.float32)
    images = list(images.unbind(0))
    return images


def load_rcnn_model(num_classes=2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    print(f"Loaded FasterRCNN model from '{CHECKPOINT_PATH}' onto {device}.")

    return model, device


def run_rcnn_interface( model, device, preprocessed_image_path = PREPROCESSED_IMAGE_PATH, visualize=False, score_threshold=0.85):
    """
    Runs the RCNN model on a 3D tensor of shape (128, 1, 128, 156)

    Args:
        preprocessed_image_path: path to a 3D image of shape (128, 156, 128)
        model (nn.Module): Loaded RCNN model
        device (torch.device): Device to run inference on
        visualize (bool): Whether to visualize the slices and bounding boxes
        score_threshold (float): Threshold for detection confidence

    Returns:
        List[List[Tensor]]: One list per slice, each containing N (4,) tensors for valid boxes
    """
    # Load preprocessed 3D numpy image
    preprocessed_image = np.load(preprocessed_image_path)
    image_list = prepare_images(preprocessed_image)
    image_slices = [img.to(device) for img in image_list]

    all_boxes = []

    for idx, img in enumerate(image_slices):
        with torch.no_grad():
            prediction = model([img])[0]

        print(f"Boxes: {prediction['boxes']}")
        print(f"Scores: {prediction['scores']}")
        boxes = prediction['boxes'].cpu()
        scores = prediction['scores'].cpu()
        labels = prediction['labels'].cpu()

        # Filter: label == 1 and score >= threshold
        keep = (labels == 1) & (scores >= score_threshold)
        filtered_boxes = boxes[keep]

        all_boxes.append([box for box in filtered_boxes])

        if visualize:
            img_np = img.cpu().squeeze().numpy()
            plt.figure(figsize=(5, 5))
            plt.imshow(img_np, cmap='gray')
            ax = plt.gca()

            for box in filtered_boxes:
                x_min, y_min, x_max, y_max = box.numpy().astype(int)
                rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                 edgecolor='red', facecolor='none', linewidth=2)
                ax.add_patch(rect)

            plt.title(f"Slice {idx} — {len(filtered_boxes)} boxes > {score_threshold}")
            plt.axis('off')
            plt.savefig(f"temp_results\img_with_boxes\slice_{idx}", bbox_inches='tight')
            print(f"saved to temp_results\img_with_boxes\slice_{idx}")

    print("rcnn done")
    return all_boxes

# model, device = load_rcnn_model()
# run_rcnn_interface("temp_results/preprocessed_slices.npy", model, device, visualize=True)