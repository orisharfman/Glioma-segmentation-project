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
from skimage.measure import label, regionprops

from torch import nn, optim
import torch.nn.functional as F
import evaluation_functions
import segmentation_models_pytorch as smp
from GeneralFunctions import TimestampToString
from monai.losses import DiceCELoss

from monai.losses import HausdorffDTLoss
import warnings

NUMPOINTSPERBOX = 20

def dice_loss(logits, targets, smooth=1.0):
    probs = torch.sigmoid(logits)
    targets = targets.float()
    intersection = (probs * targets).sum(dim=(1,2,3))
    union = probs.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


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
    medsam_seg = (low_res_pred > 0.9).astype(np.uint8)
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

def prepare_targrets(mask):
    labeled = label(mask)
    props = regionprops(labeled)
    boxes = []
    areas = []
    sampled_points = []

    if len(props) > 0:
        for prop in props:
            min_row, min_col, max_row, max_col = prop.bbox

            #potentially add a bit of padding
            area = (max_col-min_col)*(max_row-min_row)
            if area < 100:
                continue
            boxes.append([min_col, min_row, max_col, max_row])
            areas.append((max_col-min_col)*(max_row-min_row))
            # print(f"slice {z} box - {min_col, min_row, max_col, max_row}")
            min_col = max(0, min_col)
            max_col = min(mask.shape[1], max_col)
            min_row = max(0, min_row)
            max_row = min(mask.shape[0], max_row)
            width = max_col - min_col
            height = max_row - min_row
            xs = np.random.randint(min_col, max_col, size=NUMPOINTSPERBOX)
            ys = np.random.randint(min_row, max_row, size=NUMPOINTSPERBOX)
            points = list(zip(xs, ys))  # list of (x, y) tuples
            sampled_points.extend(points)

    return boxes, sampled_points, areas


def plot_image_mask_prediction(image, mask, pred):
    # Convert image from (3, H, W) to (H, W, 3)
    image = image.transpose(1, 2, 0)
    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow((image*255).astype('uint8'))
    axs[0].set_title("Image")
    axs[0].axis('off')
    axs[1].imshow(mask, cmap='gray')
    axs[1].set_title("Ground Truth Mask")
    axs[1].axis('off')
    pred = pred[0,0,:,:]
    axs[2].imshow(pred, cmap='gray')
    axs[2].set_title("Predicted Mask")
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()




MedSAM_CKPT_PATH = "Checkpoints/medsam_vit_b.pth"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"cuda available = {torch.cuda.is_available()}")
medsam_model = sam_model_registry['vit_b'](checkpoint = None)


state_dict = torch.load(MedSAM_CKPT_PATH, map_location='cpu')
medsam_model.load_state_dict(state_dict)

medsam_model = medsam_model.to(device)

for param in medsam_model.parameters():
    param.requires_grad = True
# Freeze image encoder (SAM backbone)
for param in medsam_model.image_encoder.parameters():
    param.requires_grad = False

warnings.filterwarnings("ignore", message="single channel prediction, `include_background=False` ignored.")

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    hausdorff_loss = HausdorffDTLoss()

dice_ce_loss = DiceCELoss(sigmoid = True, squared_pred = True, reduction= 'mean')

optimizer = optim.Adam(medsam_model.parameters(), lr=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

h5_data_train = "../Data/train_data.h5"
h5_masks_train = "../Data/train_masks.h5"
h5_bbox_train = "../Data/train_boxs.h5"
h5_data_val = "../Data/val_data.h5"
h5_masks_val = "../Data/val_masks.h5"
h5_bbox_val = "../Data/val_boxs.h5"
h5_data_test = "../Data/test_data.h5"
h5_masks_test = "../Data/test_masks.h5"
h5_bbox_test = "../Data/test_boxs.h5"



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


num_epochs = 30

scaler = torch.cuda.amp.GradScaler()

dice_arr_train = []
iou_arr_train = []
confusion_matrix_arr_train = []
recall_arr_train = []
precision_arr_train = []
loss_arr_train = []
dice_arr_val = []
iou_arr_val = []
confusion_matrix_arr_val = []
recall_arr_val = []
precision_arr_val = []
loss_arr_val = []


print(f"{TimestampToString()}: start training, train_size = {train_size}, val_size = {val_size}")

for epoch in range(num_epochs):
    medsam_model.train()
    train_loss = 0.0
    iou_train = 0
    dice_train = 0
    train_counter = 0
    tn_train = 0
    fp_train = 0
    fn_train = 0
    tp_train = 0
    for key_idx in range(len(image_train_keys)):
        if image_train_keys[key_idx].split("_")[1] != masks_train_keys[key_idx].split("_")[1]:  # sanity for keys
            print(f"not matching keys! {image_train_keys[key_idx]}~{masks_train_keys[key_idx]}")
            exit(-1)
        images = f_image_train[image_train_keys[key_idx]]
        masks = f_masks_train[masks_train_keys[key_idx]]
        boxes_pre = f_bbox_train[bbox_train_keys[key_idx]]

        H, W, _ = images.shape
        for i in range(0, images.shape[2]):
            # image = prepare_image(images[:,:,i])

            image = images[:,:,i]
            image = np.stack([image] * 3, axis=0)# convert the shape to (3, H, W)
            image = (image - image.min()) / np.clip(image.max() - image.min(), 1e-8, None)

            mask = masks[:,:,i,1] + masks[:,:,i,2]
            boxes , sampled_points, areas= prepare_targrets(mask)
            if len(boxes) == 0:
                continue # no box => no segmentation

            # print(areas)
            sampled_points = torch.tensor(sampled_points, dtype=torch.float32) / torch.tensor([W, H], dtype=torch.float32) * 1024
            boxes_resize = torch.tensor(boxes, dtype=torch.float32) / torch.tensor([W, H, W, H], dtype=torch.float32) * 1024

            batch = {}
            batch['image'] = torch.tensor(image, dtype=torch.float32).to(device)  # shape: (3, H, W)
            batch['original_size'] = (H, W)
            batch['boxes'] = boxes_resize.clone().detach().to(dtype=torch.float32, device=device)
            batch['points'] = sampled_points.clone().detach().to(dtype=torch.float32, device=device)

            with torch.set_grad_enabled(True):
                optimizer.zero_grad()
                # image_embedding = medsam_model.image_encoder(image)  # (1, 256, 64, 64)
                # medsam_seg = medsam_inference(medsam_model, image_embedding, box, H, W)

                medsam_seg = medsam_model([batch], multimask_output = False)  # Adjust if MedSAM expects different args


                logits = medsam_seg[0]['low_res_logits']  # shape (1, 1, 256, 256)
                logits_up = F.interpolate(logits, size=mask.shape[-2:], mode='bilinear', align_corners=False)
                pred = torch.max(logits_up, dim=0, keepdim=True)[0]
                probabilities = torch.sigmoid(pred)

                # Convert the probabilities to a binary mask (True/False) using a threshold, e.g., 0.5
                binary_mask = probabilities > 0.75

                # Count how many True values (foreground) in the binary mask
                num_true = binary_mask.sum().item()
                masks_t = []
                for box_index in range(len(boxes)):
                    box = boxes[box_index]
                    mask_i = np.zeros(mask.shape)
                    mask_i[box[1]:box[3],box[0]:box[2]] = mask[box[1]:box[3],box[0]:box[2]]
                    masks_t.append(mask_i)
                masks_np = np.array(masks_t)  # Fast conversion to single array
                masks_t = torch.tensor(masks_np, dtype=torch.float32).unsqueeze(1).to(device)
                train_counter+=1
                bce = F.binary_cross_entropy_with_logits(logits_up, masks_t.float())
                dice = dice_loss(logits_up, masks_t)
                loss = sum(areas) / 2000 * (dice + bce)
                # loss = F.binary_cross_entropy_with_logits(logits_up, masks_t.float())
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss+=loss.item()
                probabilities = probabilities.cpu().detach().numpy()
                iou_train += evaluation_functions.iou(probabilities, mask, threshold=0.75)
                dice_train += evaluation_functions.dice(probabilities, mask, threshold=0.75)
                tn, fp, fn, tp = evaluation_functions.confusion_matrix_calculation(probabilities, mask, threshold=0.75)
                tn_train+= tn
                fp_train+= fp
                fn_train+= fn
                tp_train+= tp
                # plot_image_mask_prediction(image, mask, binary_mask.cpu())

                # loss.backward()
                # optimizer.step()
    scheduler.step()
    train_loss /= train_counter
    dice_train /= train_counter
    iou_train /= train_counter

    dice_arr_train.append(dice_train)
    iou_arr_train.append(iou_train)
    confusion_matrix_arr_train.append([tn_train,fp_train,fn_train,tp_train])
    precision_arr_train.append(evaluation_functions.precision(tp_train, fp_train))
    recall_arr_train.append(evaluation_functions.recall(tp_train, fn_train))
    loss_arr_train.append(train_loss)

    np.savetxt("Meas3/iou_arr_train_0.txt", iou_arr_train)
    np.savetxt("Meas3/dice_arr_train_0.txt", dice_arr_train)
    np.savetxt("Meas3/confusion_matrix_arr_train_0.txt", confusion_matrix_arr_train)
    np.savetxt("Meas3/recall_arr_train_0.txt", recall_arr_train)
    np.savetxt("Meas3/precision_arr_train_0.txt", precision_arr_train)
    np.savetxt("Meas3/loss_arr_train_0.txt", loss_arr_train)




    val_loss = 0.0
    iou_val = 0
    dice_val = 0
    val_counter = 0
    tn_val = 0
    fp_val = 0
    fn_val = 0
    tp_val = 0
    medsam_model.eval()

    print(f"{TimestampToString()}: validation")

    for key_idx in range(len(image_val_keys)):
        if image_val_keys[key_idx].split("_")[1] != masks_val_keys[key_idx].split("_")[1]:  # sanity for keys
            print(f"not matching keys! {image_val_keys[key_idx]}~{masks_val_keys[key_idx]}")
            exit(-1)
        images = f_image_val[image_val_keys[key_idx]]
        masks = f_masks_val[masks_val_keys[key_idx]]
        boxes_pre = f_bbox_val[bbox_val_keys[key_idx]]

        H, W, _ = images.shape
        for i in range(0, images.shape[2]):
            # image = prepare_image(images[:,:,i])

            image = images[:,:,i]
            image = np.stack([image] * 3, axis=0)# convert the shape to (3, H, W)
            image = (image - image.min()) / np.clip(image.max() - image.min(), 1e-8, None)

            mask = masks[:,:,i,1] + masks[:,:,i,2]
            boxes, sampled_points, areas= prepare_targrets(mask)

            if len(boxes) == 0:
                continue # no box => no segmentation
            sampled_points = torch.tensor(sampled_points, dtype=torch.float32) / torch.tensor([W, H], dtype=torch.float32) * 1024
            boxes_resize = torch.tensor(boxes, dtype=torch.float32) / torch.tensor([W, H, W, H], dtype=torch.float32) * 1024
            batch = {}
            batch['image'] = torch.tensor(image, dtype=torch.float32).to(device)  # shape: (3, H, W)
            batch['original_size'] = (H, W)
            batch['boxes'] = boxes_resize.clone().detach().to(dtype=torch.float32, device=device)
            batch['points'] = sampled_points.clone().detach().to(dtype=torch.float32, device=device)


            with torch.no_grad():
                # image_embedding = medsam_model.image_encoder(image)  # (1, 256, 64, 64)
                # medsam_seg = medsam_inference(medsam_model, image_embedding, box, H, W)

                medsam_seg = medsam_model([batch], multimask_output = False)  # Adjust if MedSAM expects different args


                logits = medsam_seg[0]['low_res_logits']  # shape (1, 1, 256, 256)
                logits_up = F.interpolate(logits, size=mask.shape[-2:], mode='bilinear', align_corners=False)
                pred = torch.max(logits_up, dim=0, keepdim=True)[0]
                probabilities = torch.sigmoid(pred)

                # Convert the probabilities to a binary mask (True/False) using a threshold, e.g., 0.5
                binary_mask = probabilities > 0.75

                # Count how many True values (foreground) in the binary mask
                num_true = binary_mask.sum().item()
                masks_t = []
                for box_index in range(len(boxes)):
                    box = boxes[box_index]
                    mask_i = np.zeros(mask.shape)
                    mask_i[box[1]:box[3],box[0]:box[2]] = mask[box[1]:box[3],box[0]:box[2]]
                    masks_t.append(mask_i)
                masks_np = np.array(masks_t)  # Fast conversion to single array
                masks_t = torch.tensor(masks_np, dtype=torch.float32).unsqueeze(1).to(device)
                val_counter+=1
                bce = F.binary_cross_entropy_with_logits(logits_up, masks_t.float())
                dice = dice_loss(logits_up, masks_t)

                loss = sum(areas) / 2000 * (dice + bce)
                loss = sum(areas) / 2000 * (dice_loss(logits_up, masks_t) + 0.01 * hausdorff_loss(torch.sigmoid(logits_up), masks_t.float()))
                # loss = F.binary_cross_entropy_with_logits(logits_up, masks_t.float())
                optimizer.zero_grad()
                val_loss+=loss.item()
                probabilities = probabilities.cpu().numpy()
                iou_val += evaluation_functions.iou(probabilities, mask, threshold=0.75)
                dice_val += evaluation_functions.dice(probabilities, mask, threshold=0.75)
                tn, fp, fn, tp = evaluation_functions.confusion_matrix_calculation(probabilities, mask, threshold=0.75)
                tn_val+= tn
                fp_val+= fp
                fn_val+= fn
                tp_val+= tp



    torch.save(medsam_model.state_dict(), f"Checkpoints3/medsam_weights_0_{int(epoch + 1)}.pth")

    val_loss /= val_counter
    dice_val /= val_counter
    iou_val /= val_counter

    dice_arr_val.append(dice_val)
    iou_arr_val.append(iou_val)
    confusion_matrix_arr_val.append([tn_val, fp_val, fn_val, tp_val])
    precision_arr_val.append(evaluation_functions.precision(tp_val, fp_val))
    recall_arr_val.append(evaluation_functions.recall(tp_val, fn_val))
    loss_arr_val.append(val_loss)

    np.savetxt("Meas3/iou_arr_val_0.txt", iou_arr_val)
    np.savetxt("Meas3/dice_arr_val_0.txt", dice_arr_val)
    np.savetxt("Meas3/confusion_matrix_arr_val_0.txt", confusion_matrix_arr_val)
    np.savetxt("Meas3/recall_arr_val_0.txt", recall_arr_val)
    np.savetxt("Meas3/precision_arr_val_0.txt", precision_arr_val)
    np.savetxt("Meas3/loss_arr_val_0.txt", loss_arr_val)

    print(f"{TimestampToString()}: Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")









