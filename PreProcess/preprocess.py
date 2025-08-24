# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import nibabel as nib
import numpy as np
import glob
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

TRAIN_DATASET_PATH = 'C:\\project_dataset\\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
# im_num = '00002-000'
# im_type = 't1c'
# test_image_t1c = nib.load(f'{TRAIN_DATASET_PATH}\\BraTS-GLI-{im_num}\\BraTS-GLI-{im_num}-{im_type}.nii.gz').get_fdata()
# print(test_image_t1c.max())
# print(test_image_t1c.shape)
#
# test_image_t1c = scaler.fit_transform(test_image_t1c.reshape(-1, test_image_t1c.shape[-1])).reshape(
#     test_image_t1c.shape)
# print(test_image_t1c.max())
# print(test_image_t1c.shape)
#
# im_type = 't1n'
# test_image_t1n = nib.load(f'{TRAIN_DATASET_PATH}\\BraTS-GLI-{im_num}\\BraTS-GLI-{im_num}-{im_type}.nii.gz').get_fdata()
# test_image_t1n = scaler.fit_transform(test_image_t1n.reshape(-1, test_image_t1n.shape[-1])).reshape(
#     test_image_t1n.shape)
#
# im_type = 't2f'
# test_image_t2f = nib.load(f'{TRAIN_DATASET_PATH}\\BraTS-GLI-{im_num}\\BraTS-GLI-{im_num}-{im_type}.nii.gz').get_fdata()
# test_image_t2f = scaler.fit_transform(test_image_t2f.reshape(-1, test_image_t2f.shape[-1])).reshape(
#     test_image_t2f.shape)
#
# im_type = 't2w'
# test_image_t2w = nib.load(f'{TRAIN_DATASET_PATH}\\BraTS-GLI-{im_num}\\BraTS-GLI-{im_num}-{im_type}.nii.gz').get_fdata()
# test_image_t2w = scaler.fit_transform(test_image_t2w.reshape(-1, test_image_t2w.shape[-1])).reshape(
#     test_image_t2w.shape)
#
# im_type = 'seg'
# test_image_seg = nib.load(f'{TRAIN_DATASET_PATH}\\BraTS-GLI-{im_num}\\BraTS-GLI-{im_num}-{im_type}.nii.gz').get_fdata()
# test_image_seg = test_image_seg.astype(np.uint8)
#
# print("seg")
#
# print(test_image_seg.max())
# print(test_image_seg.shape)
# print(np.unique(test_image_seg))
#
# import random
#
# n = random.randint(0, test_image_seg.shape[2])
# n = 66 + 3
# print(f'n={n}')
# plt.figure(figsize=(12, 8))
#
# plt.subplot(231)
# plt.imshow(test_image_t1c[:, :, n], cmap='gray')
# plt.title("t1c scan")
#
# plt.subplot(232)
# plt.imshow(test_image_t1n[:, :, n], cmap='gray')
# plt.title("t1n scan")
#
# plt.subplot(233)
# plt.imshow(test_image_t2f[:, :, n], cmap='gray')
# plt.title("t2f scan")
#
# plt.subplot(234)
# plt.imshow(test_image_t2w[:, :, n], cmap='gray')
# plt.title("t2w scan")
#
# plt.subplot(235)
# plt.imshow(test_image_seg[:, :, n])
# plt.title("mask")
#
# plt.show()
#
# combined_x = np.stack([test_image_t1c, test_image_t2f, test_image_t2w], axis=3)
# combined_x = combined_x[56:184, 56:184, 13:141]
#
# test_image_seg = test_image_seg[56:184, 56:184, 13:141]
#
# plt.figure(figsize=(12, 8))
# n = n - 13
# plt.subplot(231)
# plt.imshow(combined_x[:, :, n, 0], cmap='gray')
# plt.title("preprocessed t1c")
#
# plt.subplot(232)
# plt.imshow(combined_x[:, :, n, 1], cmap='gray')
# plt.title("preprocessed t2f")
#
# plt.subplot(233)
# plt.imshow(combined_x[:, :, n, 2], cmap='gray')
# plt.title("preprocessed t2w")
#
# plt.subplot(235)
# plt.imshow(test_image_seg[:, :, n])
# plt.title("preprocessed mask")
#
# plt.show()

# np.save(f'TrainingData\\combined_{im_num}.npy',combined_x)

# img = np.load(f'TrainingData\\combined_{im_num}.npy')
# mask = to_categorical(test_image_seg,num_classes=4)

def PreprocessT1C(InputPath , OutputPath):
    img_t1c = nib.load(InputPath).get_fdata()
    print(img_t1c.shape)
    img_t1c = scaler.fit_transform(img_t1c.reshape(-1, img_t1c.shape[-1])).reshape(img_t1c.shape)
    img_t1c = img_t1c[56: 184, 44: 200, 13: 141]
    print(img_t1c.shape)
    np.save(OutputPath, img_t1c)

t1c_list = sorted(glob.glob(f'{TRAIN_DATASET_PATH}\\BraTS-GLI-*\\BraTS-GLI-*-t1c.nii.gz'))
t1n_list = sorted(glob.glob(f'{TRAIN_DATASET_PATH}\\BraTS-GLI-*\\BraTS-GLI-*-t1n.nii.gz'))
t2f_list = sorted(glob.glob(f'{TRAIN_DATASET_PATH}\\BraTS-GLI-*\\BraTS-GLI-*-t2f.nii.gz'))
t2w_list = sorted(glob.glob(f'{TRAIN_DATASET_PATH}\\BraTS-GLI-*\\BraTS-GLI-*-t2w.nii.gz'))
mask_list = sorted(glob.glob(f'{TRAIN_DATASET_PATH}\\BraTS-GLI-*\\BraTS-GLI-*-seg.nii.gz'))

print(len(t1c_list))
print(len(t1n_list))
print(len(t2f_list))
print(len(t2w_list))
print(len(mask_list))

for i in range(len(t1c_list)):
    PreprocessT1C(t1c_list[i],"output.npy")
    exit(0)
    img_t1c = nib.load(t1c_list[i]).get_fdata()
    img_t1c = scaler.fit_transform(img_t1c.reshape(-1, img_t1c.shape[-1])).reshape(img_t1c.shape)

    img_t1n = nib.load(t1n_list[i]).get_fdata()
    img_t1n = scaler.fit_transform(img_t1n.reshape(-1, img_t1n.shape[-1])).reshape(img_t1n.shape)

    img_t2f = nib.load(t2f_list[i]).get_fdata()
    img_t2f = scaler.fit_transform(img_t2f.reshape(-1, img_t2f.shape[-1])).reshape(img_t2f.shape)

    img_t2w = nib.load(t2w_list[i]).get_fdata()
    img_t2w = scaler.fit_transform(img_t2w.reshape(-1, img_t2w.shape[-1])).reshape(img_t2w.shape)

    img_mask = nib.load(mask_list[i]).get_fdata()
    img_mask = img_mask.astype(np.uint8)

    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.imshow(img_t1c[:, :, 66],cmap = 'gray')
    plt.title("t1c scan")

    plt.subplot(232)
    plt.imshow(img_t1n[:, :, 66],cmap = 'gray')
    plt.title("t1n scan")

    plt.subplot(233)
    plt.imshow(img_t2f[:, :, 66],cmap = 'gray')
    plt.title("t2f scan")

    plt.subplot(234)
    plt.imshow(img_t2w[:, :, 66],cmap = 'gray')
    plt.title("t2w scan")

    plt.subplot(235)
    plt.imshow(img_mask[:, :, 66])
    plt.title("mask")

    plt.show()

    img_mask = nib.load(mask_list[i]).get_fdata()
    img_mask = img_mask.astype(np.uint8)
    img_mask[img_mask == 2] = 0
    img_mask[img_mask == 3] = 2
    img_combined = img_t1c
    img_combined = img_combined[56:184, 44:200, 13:141]
    plt.imshow(img_combined[:, :, 66], cmap='gray')
    img_mask = img_mask[56:184, 44:200, 13:141]

    #display all for debug
    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.imshow(img_combined[:, :, 66-13],cmap = 'gray')
    plt.title("t1c scan")

    plt.subplot(232)
    plt.imshow(img_mask[:, :, 66-13])
    plt.title("mask")

    plt.tight_layout()
    plt.show()
    exit(0)

    val, counts = np.unique(img_mask, return_counts=True)

    if 1 - counts[0] / counts.sum() > 0.01:
        print(f"save image {i}")
        img_mask = to_categorical(img_mask, num_classes=3)
        np.save(f"TrainingData_t1c\\images\\image_{i}.npy", img_combined)
        np.save(f"TrainingData_t1c\\masks\\mask_{i}", img_mask)
    else:
        print(f'skipping image {i}')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
