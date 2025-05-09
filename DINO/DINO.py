import torch
from torchvision import transforms as pth_transforms
from PIL import Image
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

feature_map = None
def hook_fn(module, input, output):
    global feature_map
    feature_map = output[0].detach()


def prepare_images(images):

    images = resize(images, (300, 300, 128), anti_aliasing=True, mode="reflect")
    images = np.transpose(images, (
        2, 0, 1))  # Shape: (128, 300, 300) - reorder so index 0 will be the layer (instead of 2)
    images = np.expand_dims(images, axis=1)  # Shape: (128, 1, 300, 300)
    images = torch.tensor(images, dtype=torch.float32)
    images = list(images.unbind(0))
    return images


def prepare_targrets(masks):
    masks = resize(masks, (300, 300, 128), anti_aliasing=True, mode="reflect")
    targets = []  # input as list of dictionaries
    for z in range(masks.shape[2]):  # for each slice
        slice_2d = masks[:, :, z]
        rows, cols = np.where(slice_2d > 0)
        if rows.size > 0 and cols.size > 0:
            x_min, x_max = cols.min() - 3, cols.max() + 3
            y_min, y_max = rows.min() - 3, rows.max() + 3
            targets.append({
                "boxes": torch.tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.int64),
                "area": torch.tensor([(x_max - x_min) * (y_max - y_min)], dtype=torch.float32),
                "iscrowd": torch.empty((0,), dtype=torch.int64),
            })
        else:
            # If the slice has no non-zero elements, add empty list
            targets.append({
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.empty((0,), dtype=torch.int64),
                "area": torch.empty((0,), dtype=torch.float32),
                "iscrowd": torch.empty((0,), dtype=torch.int64),
            })
    return targets




model_dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True)
hook = model_dino.blocks[-1].attn.register_forward_hook(hook_fn)
model_dino.eval()

#if GPU is available
device = torch.device("cuda")
model_dino.to(device)

h5_data_train = "../Data/train_data.h5"
h5_masks_train = "../Data/train_masks.h5"
h5_data_val = "../Data/val_data.h5"
h5_masks_val = "../Data/val_masks.h5"
h5_data_test = "../Data/test_data.h5"
h5_masks_test = "../Data/test_masks.h5"

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


transform = pth_transforms.Compose([
    pth_transforms.ToTensor(),
    pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


img3d = f_image_train[image_train_keys[0]]
print(image_train_keys[0])
embeddings = []
for i in range(128):
    if i != 66: continue
    print(f"process {i}")
    img = img3d[:,:,i]
    img = np.stack([img]*3, axis=-1)
    img_tensor = transform(img)
    img_tensor = img_tensor.float()
    img_tensor = img_tensor.unsqueeze(0).cuda()
    with torch.no_grad():
        emb = model_dino(img_tensor)
        # embeddings.append(emb.squeeze().cpu().numpy())
    feature_map = feature_map.squeeze().cpu().numpy()  # Remove batch dimension and convert to numpy
    # Plotting both the original image and the feature map side by side
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))

    # Original Image
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')  # Hide axes

    # Feature Map (Heatmap)
    axes[1].imshow(feature_map, cmap='viridis')
    axes[1].set_title('Feature Map')
    axes[1].axis('off')  # Hide axes

    # Show the plot
    plt.tight_layout()
    plt.show()n

embeddings = np.array(embeddings)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)
plt.figure(figsize=(8, 6))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=40, c='blue', alpha=0.7)
plt.title("DINO Embeddings (t-SNE Visualization)")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.grid(True)
plt.show()