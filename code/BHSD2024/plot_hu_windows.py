import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def apply_window(img, wl, ww):
    """
    Apply HU windowing to a CT image.
    wl: window level (center)
    ww: window width
    """
    min_val = wl - ww / 2.0
    max_val = wl + ww / 2.0
    img_windowed = np.clip(img, min_val, max_val)
    img_windowed = (img_windowed - min_val) / (max_val - min_val)
    return img_windowed

def main():
    image_dir = "/mnt/d/BHSD/label_192/images"
    target_filename = "ID_b1d5d5e8_ID_4cfa36b63f.nii.gz"
    filepath = os.path.join(image_dir, target_filename)

    if not os.path.exists(filepath):
        print(f"Error: File {filepath} does not exist.")
        return

    # Define windows (Level, Width) in the requested order: Bone, Subdural, Brain
    windows = {
        "Bone Window\n(WL=600, WW=2800)": (600, 2800),
        "Subdural Window\n(WL=75, WW=130)": (75, 130),
        "Brain Window\n(WL=40, WW=80)": (40, 80)
    }

    print(f"Processing {target_filename}...")
    
    # Load NIfTI file
    nii_img = nib.load(filepath)
    data = nii_img.get_fdata()
    
    # Extract middle slice (along the last dimension, depth)
    mid_idx = data.shape[-1] // 2
    if len(data.shape) == 3:
        slice_data = data[:, :, mid_idx]
    elif len(data.shape) == 4:
        slice_data = data[:, :, mid_idx, 0]
    else:
        print(f"Unexpected image shape: {data.shape}")
        return

    # Rotate 180 degrees compared to the original orientation
    slice_data = np.rot90(slice_data, k=3)

    # Plot 3 windows side-by-side (without suptitle)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (title, (wl, ww)) in zip(axes, windows.items()):
        windowed_slice = apply_window(slice_data, wl, ww)
        ax.imshow(windowed_slice, cmap='gray', origin='lower')
        ax.set_title(title, fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    save_path = f"/home/thanh/SWDL/hu_windows_ID_b1d5d5e8_ID_4cfa36b63f.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    main()
