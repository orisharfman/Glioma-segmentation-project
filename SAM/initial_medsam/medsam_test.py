# %% environment and functions
from lib2to3.fixes.fix_metaclass import FixMetaclass

import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import sys
join = os.path.join
import torch
from segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import evaluation_functions

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))

@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :] # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed, # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
        multimask_output=False,
        )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg


# %% image preprocessing and model inference
def prepare_image(img_2d):
    img_1024 = transform.resize(img_2d, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(
        np.float32)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    )  # normalize to [0, 1], (H, W, 3)\

    # convert the shape to (3, H, W)
    img_1024 = np.stack([img_1024] * 3, axis=0)
    img_1024_tensor = torch.tensor(img_1024).float()

    # img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
    return img_1024_tensor

MedSAM_CKPT_PATH = "medsam_vit_b.pth"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"cuda available = {torch.cuda.is_available()}")
medsam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)



medsam_model.eval()

h5_data_train = "../Data/train_data.h5"
h5_masks_train = "../Data/train_masks.h5"
h5_bbox_train = "../Data/train_boxs.h5"
h5_data_val = "../Data/val_data.h5"
h5_masks_val = "../Data/val_masks.h5"
h5_bbox_val = "../Data/val_boxs.h5"
h5_data_test = "../Data/test_data.h5"
h5_masks_test = "../Data/test_masks.h5"
h5_bbox_test = "../Data/test_boxs.h5"

dice_arr = []
iou_arr = []
confusion_matrix_arr = []
recall_arr = []
precision_arr = []

f_image_train = h5py.File(h5_data_train, "r")
f_masks_train = h5py.File(h5_masks_train, "r")
f_bbox_train = h5py.File(h5_bbox_train, "r")
f_image_val = h5py.File(h5_data_val, "r")
f_masks_val = h5py.File(h5_masks_val, "r")
f_bbox_val = h5py.File(h5_bbox_val, "r")
image_train_keys = sorted(list(f_image_train.keys()))
masks_train_keys = sorted(list(f_masks_train.keys()))
bbox_train_keys = sorted(list(f_bbox_train.keys()))

image_val_keys = sorted(list(f_image_val.keys()))
masks_val_keys = sorted(list(f_masks_val.keys()))
bbox_val_keys = sorted(list(f_bbox_val.keys()))

train_size = len(image_train_keys)
val_size = len(image_val_keys)


num_epochs = 10

for epoch in range(num_epochs):
    medsam_model.train()
    running_loss = 0.0



    for key_idx in range(len(image_train_keys)):
        if image_train_keys[key_idx].split("_")[1] != masks_train_keys[key_idx].split("_")[1]:  # sanity for keys
            print(f"not matching keys! {image_train_keys[key_idx]}~{masks_train_keys[key_idx]}")
            exit(-1)
        images = f_image_train[image_train_keys[key_idx]]
        masks = f_masks_train[masks_train_keys[key_idx]]
        boxes = f_bbox_train[bbox_train_keys[key_idx]]

        H, W, _ = images.shape
        for i in range(0, images.shape[2]):
            image = prepare_image(images[:,:,i]).unsqueeze(0).to(device)

            box = boxes[i]
            box  = box * 1024 / np.array([W,H,W,H])
            box = np.expand_dims(box, axis=0)  # (4,) → (1, 4)

            mask = masks[:,:,i,1]
            box = box.to(device)
            mask = mask.to(device)

            with torch.set_grad_enabled(True):
                image_embedding = medsam_model.image_encoder(image)  # (1, 256, 64, 64)
                medsam_seg = medsam_inference(medsam_model, image_embedding, box, H, W)
                loss = criterion(medsam_seg, mask.float())
                loss.backward()
                optimizer.step()

        #check if mask exists
        iou_arr.append(evaluation_functions.iou(medsam_seg, mask, threshold=0.5))
        dice_arr.append(evaluation_functions.dice(medsam_seg, mask, threshold=0.5))
        tn, fp, fn, tp = evaluation_functions.confusion_matrix_calculation(medsam_seg, mask, threshold=0.5)
        precision_arr.append(evaluation_functions.precision(tp, fp))
        recall_arr.append(evaluation_functions.recall(tp, fn))
np.savetxt("initial_medsam/Meas/iou_arr.txt", iou_arr)
np.savetxt("initial_medsam/Meas//dice_arr.txt", dice_arr)
np.savetxt("initial_medsam/Meas//recall_arr.txt", recall_arr)
np.savetxt("initial_medsam/Meas//precision_arr.txt", precision_arr)


















