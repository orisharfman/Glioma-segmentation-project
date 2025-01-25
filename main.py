import numpy as np
import torch
import h5py
h5_masks = "Data/masks.h5"

if __name__ == '__main__':
    print(torch.__version__)
    if torch.cuda.is_available():
        print("yes!")
        print(f"CUDA version {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    with h5py.File('Data/images.h5', 'r') as img5f:
        print(img5f.keys())
        img = img5f['image_0']
        print(img.shape)
        print(img.dtype)
        img = img[:,:,64:]
        img = np.transpose(img, (2, 0, 1))
        slices = np.expand_dims(img, axis=1)
        print(slices.shape)
        print(slices.max())
        print(slices.min())

    with h5py.File(h5_masks, 'r') as f:
        masks_keys = [key for key in f.keys()]

        print(masks_keys[0])
        mask = f[masks_keys[0]][:]  # Shape: (128, 156, 128, 2)
        mask = mask[:, :, :, 1].astype(np.uint8)  # Shape: (128, 156, 128)
        print(mask.shape)
        print(mask.dtype)
        print(mask.max())
        bounding_boxes = []
        for z in range(mask.shape[2]):
            slice_2d = mask[:, :, z]
            rows, cols = np.where(slice_2d > 0)
            if rows.size > 0 and cols.size > 0:
                x_min, x_max = cols.min(), cols.max()
                y_min, y_max = rows.min(), rows.max()
                bounding_boxes.append([x_min, y_min, x_max, y_max])
            else:
                # If the slice has no non-zero elements, add a default box ((-1, -1, -1, -1))
                bounding_boxes.append([-1, -1, -1, -1])
        print(bounding_boxes)
        mask = np.array(bounding_boxes)  # Shape: (128,4)
        print(mask.shape)
        print(mask.dtype)
        print(mask.max())

