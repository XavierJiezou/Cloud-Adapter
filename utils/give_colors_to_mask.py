import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Define the function to retrieve the color palette for a given dataset
def get_palette(dataset_name: str):
    if dataset_name in ["cloudsen12_high_l1c", "cloudsen12_high_l2a"]:
        return [79, 253, 199, 77, 2, 115, 251, 255, 41, 221, 53, 223]
    if dataset_name == "l8_biome":
        return [79, 253, 199, 221, 53, 223, 251, 255, 41, 77, 2, 115]
    if dataset_name in ["gf12ms_whu_gf1", "gf12ms_whu_gf2", "hrc_whu"]:
        return [79, 253, 199, 77, 2, 115]
    raise Exception("dataset_name not supported")

# Function to apply the color palette to a mask
def give_colors_to_mask(mask: np.ndarray, colors=None) -> np.ndarray:
    """Convert a mask to a colorized version using the specified palette."""
    im = Image.fromarray(mask.astype(np.uint8)).convert("P")
    im.putpalette(colors)
    return im

# Function to process a single file
def process_file(file_path, palette):
    try:
        # Load the mask
        mask = np.array(Image.open(file_path))

        # Apply the color palette
        colored_mask = give_colors_to_mask(mask, palette)

        # Save the colored mask, overwriting the original file
        colored_mask.save(file_path)
        return True
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

# Main processing function for a dataset
def process_dataset(dataset_name, base_root, progress_bar):
    ann_dir = os.path.join(base_root, dataset_name, "ann_dir")
    if not os.path.exists(ann_dir):
        print(f"Annotation directory does not exist for {dataset_name}: {ann_dir}")
        return

    # Get the color palette for this dataset
    palette = get_palette(dataset_name)

    # Gather all files to process
    files_to_process = []
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(ann_dir, split)
        if not os.path.exists(split_dir):
            print(f"Split directory does not exist for {dataset_name}: {split_dir}")
            continue

        # Add all png files in the directory to the list
        for file_name in os.listdir(split_dir):
            if file_name.endswith(".png"):
                files_to_process.append(os.path.join(split_dir, file_name))

    # Multi-threaded processing
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(lambda f: process_file(f, palette), files_to_process),
            total=len(files_to_process),
            desc=f"Processing {dataset_name}",
            leave=False
        ))

    # Update the progress bar
    progress_bar.update(len(files_to_process))

    print(f"{dataset_name}: Processed {sum(results)} files out of {len(files_to_process)}.")

# Define the root directory and datasets
base_root = "data"  # Replace with your datasets' root directory
dataset_names = [
    "cloudsen12_high_l1c",
    "cloudsen12_high_l2a",
    "gf12ms_whu_gf1",
    "gf12ms_whu_gf2",
    "hrc_whu",
    "l8_biome"
]

# Main script
if __name__ == "__main__":
    # Calculate total number of files for all datasets
    total_files = 0
    for dataset_name in dataset_names:
        ann_dir = os.path.join(base_root, dataset_name, "ann_dir")
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(ann_dir, split)
            if os.path.exists(split_dir):
                total_files += len([f for f in os.listdir(split_dir) if f.endswith(".png")])

    # Create a progress bar
    with tqdm(total=total_files, desc="Overall Progress") as progress_bar:
        for dataset_name in dataset_names:
            process_dataset(dataset_name, base_root, progress_bar)
