# %% environment and functions
# from lib2to3.fixes.fix_metaclass import FixMetaclass
#
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
join = os.path.join
import torch
from segment_anything import sam_model_registry
from skimage import io, transform
from skimage.measure import label, regionprops

from torch import nn, optim
import torch.nn.functional as F
# import evaluation_functions
import segmentation_models_pytorch as smp
# from GeneralFunctions import TimestampToString
#
#
MedSAM_CKPT_PATH = "checkpoints/medsam_weights_1_19.pth"
PREPROCESSED_IMAGE_PATH = "temp_results\preprocessed_slices.npy"



#
def prepare_images(images):
    images = np.transpose(images, (2, 0, 1))  # Shape: (128, 156, 128) - reorder so index 0 will be the layer (instead of 2)

    images = torch.tensor(images, dtype=torch.float32)
    images = list(images.unbind(0))
    return images

def load_sam_model():
    medsam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    medsam_model = medsam_model.to(device)
    medsam_model.eval()
    print(f"Loaded sam model from '{MedSAM_CKPT_PATH}' onto {device}.")

    return medsam_model, device

def run_sam_interface( model, device, rcnn_boxes, preprocessed_image_path = PREPROCESSED_IMAGE_PATH, visualize=False, score_threshold=0.99):
    """
    Runs the sam model on a 3D tensor of shape (128, 1, 128, 156)

    Args:
        preprocessed_image_path: path to a 3D image of shape (128, 156, 128)
        model (nn.Module): Loaded RCNN model
        device (torch.device): Device to run inference on
        visualize (bool): Whether to visualize the slices and masks
        score_threshold (float): Threshold for detection confidence

    Returns:
        segmentation_mask (np.ndarray): Binary mask of shape (128, 156, 128)
    """
    # Load preprocessed 3D numpy image
    preprocessed_image = np.load(preprocessed_image_path)
    image_list = prepare_images(preprocessed_image)
    slices = []
    for idx, img in enumerate(image_list):
        H, W = img.shape
        if len(rcnn_boxes[idx]) == 0:#no box => no segmentation in the slice
            slices.append(np.zeros((H,W,1)))
            plt.figure(figsize=(5, 5))
            img_np = img.cpu().squeeze().numpy()
            img_np = (img_np * 255).astype('uint8')
            plt.imshow(img_np, cmap='gray')
            #plt.title(f"Slice {idx} — {len(image_list)} mask > {score_threshold}")
            plt.axis('off')
            plt.savefig(f"temp_results\img_with_mask\slice_{idx}", bbox_inches='tight')
            print(f"saved to temp_results\img_with_mask\slice_{idx}")
            continue

        batch = {}

        img_stack = np.stack([img] * 3, axis=0)  # convert the shape to (3, H, W)
        batch['image'] = torch.tensor(img_stack, dtype=torch.float32).to(device)  # shape: (3, H, W)
        batch['original_size'] = (H, W)

        rcnn_boxes_list = [box.tolist() for box in rcnn_boxes[idx]]

        boxes_resize = torch.tensor(rcnn_boxes_list, dtype=torch.float32) / torch.tensor([W, H, W, H], dtype=torch.float32) * 1024
        batch['boxes'] = boxes_resize.to(dtype=torch.float32, device=device)

        with torch.no_grad():
            medsam_seg = model([batch], multimask_output=False)

        logits = medsam_seg[0]['low_res_logits']  # shape (1, 1, 256, 256)
        logits_up = F.interpolate(logits, size=(H,W), mode='bilinear', align_corners=False)

        pred = torch.max(logits_up, dim=0, keepdim=True)[0]

        probabilities = torch.sigmoid(pred)

        # Convert the probabilities to a binary mask (True/False) using a threshold, e.g., 0.5
        binary_mask = probabilities > score_threshold
        binary_mask = binary_mask.cpu()
        binary_mask = binary_mask[0,0,:,:]
        slices.append(binary_mask[..., np.newaxis])
        if visualize:
            img_np = img.cpu().squeeze().numpy()
            img_np = (img_np * 255).astype('uint8')
            plt.figure(figsize=(5, 5))
            plt.imshow(img_np, cmap='gray')
            overlay = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 4))  # 4 channels: R, G, B, Alpha
            overlay[..., 1] = 1.0  # Red
            overlay[..., 3] = binary_mask * 0.25  # Alpha channel, only where mask == 1
            plt.imshow(overlay)  # Red overlay with transparency
            #plt.title(f"Slice {idx} — {len(image_list)} mask > {score_threshold}")
            plt.axis('off')
            plt.savefig(f"temp_results\img_with_mask\slice_{idx}", bbox_inches='tight')
            print(f"saved to temp_results\img_with_mask\slice_{idx}")
    segmentation_mask = np.concatenate(slices, axis=2)
    np.save(f"temp_results\masks\segmentation_mask.npy",segmentation_mask)
    return segmentation_mask


def run_sam_model():
    print("in SAM")
    return