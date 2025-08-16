import numpy as np
import nibabel as nib
from scipy.spatial.distance import pdist
from scipy.ndimage import binary_erosion
from scipy.ndimage import center_of_mass
import os
import pickle



def load_mask_folder_as_3d_array(folder_path):
    """
    Loads 2D .npy mask files from a folder and stacks them into a 3D numpy array.
    Assumes files are named or sorted so the order matches slice order.

    Args:
        folder_path (str): path to folder containing 2D .npy mask files.

    Returns:
        np.ndarray: 3D mask array (height x width x num_slices)
    """

    # List all .npy files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

    if not files:
        raise ValueError(f"No .npy files found in {folder_path}")

    # Sort files by filename to maintain slice order
    files.sort()

    slices = []
    for filename in files:
        slice_path = os.path.join(folder_path, filename)
        slice_mask = np.load(slice_path)

        # Optional: ensure it's binary mask (0 or 1)
        slice_mask = (slice_mask > 0).astype(np.uint8)

        slices.append(slice_mask)

    # Stack slices along the third dimension (z-axis)
    mask_3d = np.stack(slices, axis=2)

    return mask_3d


def calculate_statistics(segmentation_mask):
    """
    Args:
        segmentation_mask: 3D numpy array (binary mask: 1 = tumor, 0 = background)
        spacing: tuple/list of voxel spacing (dx, dy, dz) in mm or cm
        brain_volume: float, total brain volume in same units (cm³) or None

    Returns:
        dict of statistics
    """
    print("calculating stats")

    nii_file = os.path.join("temp_input", os.listdir("temp_input")[0])
    nii = nib.load(nii_file)
    image = nii.get_fdata()
    spacing = nii.header.get_zooms()    # (x_spacing, y_spacing, z_spacing)
    # Convert spacing from mm to cm if needed (assuming spacing in mm here)
    voxel_volume_cm3 = (spacing[0] / 10) * (spacing[1] / 10) * (spacing[2] / 10)

    # Brain mask: nonzero voxels
    brain_voxel_count = np.count_nonzero(image)
    voxel_volume = spacing[0] * spacing[1] * spacing[2]  # in mm^3
    brain_volume_mm3 = brain_voxel_count * voxel_volume
    brain_volume = brain_volume_mm3 / 1000  # convert to cm³

    # Total tumor volume in cm³
    tumor_voxels = np.sum(segmentation_mask > 0)
    total_volume = tumor_voxels * voxel_volume_cm3
    # Number of slices with tumor (assuming slices along axis 2)
    slices_with_tumor = np.sum(np.any(segmentation_mask > 0, axis=(0,1)))
    tumor_center_of_mass = tuple(np.round(center_of_mass(segmentation_mask)).astype(int))

    # Maximum diameter: approximate max distance between any two tumor points
    # Use bounding box diagonal or max distance in coords

    # Get coordinates of tumor voxels
    coords = np.argwhere(segmentation_mask > 0)
    if coords.size == 0:
        max_diameter = 0.0
        sphericity = 0.0
    else:
        # Convert voxel indices to physical coordinates (cm)
        physical_coords = coords * np.array(spacing) / 10  # mm to cm

        # Max pairwise distance (brute force might be slow for large tumors)
        distances = pdist(physical_coords)
        max_diameter = distances.max() if len(distances) > 0 else 0.0

        # Sphericity = (pi^(1/3))*(6*V)^(2/3) / Surface Area
        # Approximate surface area using binary erosion + surface voxels


        eroded = binary_erosion(segmentation_mask)
        surface = segmentation_mask - eroded
        surface_voxels = np.sum(surface > 0)
        # Surface area approx = surface voxels * voxel face area (approximate)
        # Voxel face area: average of 3 faces (dx*dy + dy*dz + dx*dz)/3
        voxel_face_area = ((spacing[0]*spacing[1] + spacing[1]*spacing[2] + spacing[0]*spacing[2]) / 3) / 100  # mm² to cm²
        surface_area = surface_voxels * voxel_face_area

        if surface_area > 0:
            sphericity = (np.pi**(1/3)) * (6*total_volume)**(2/3) / surface_area
            sphericity = min(max(sphericity, 0.0), 1.0)  # Clamp between 0 and 1
        else:
            sphericity = 0.0

    # Percentage of brain volume affected
    if brain_volume and brain_volume > 0:
        percent_brain_affected = 100 * total_volume / brain_volume
    else:
        percent_brain_affected = None

    stats =  {
        "Total Tumor Volume (cm³)": f"{total_volume:.2f}",
        "Max Diameter (cm)": f"{max_diameter:.2f}",
        "Tumor Center Of Mass" : f"{tumor_center_of_mass}",
        "Sphericity": f"{sphericity:.3f}",
        "Slices with Tumor": slices_with_tumor,
        "Percent Brain Volume": f"{percent_brain_affected:.2f}" if percent_brain_affected is not None else "N/A",
    }
    with open(f"temp_results\stats\stats.pkl", "wb") as f:
        pickle.dump(stats, f)
    return stats

