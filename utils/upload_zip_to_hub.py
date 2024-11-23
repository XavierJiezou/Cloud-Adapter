import os
import zipfile
from huggingface_hub import HfApi, HfFolder

# Define the root directory containing all datasets
base_root = "data"  # Replace with the directory containing all datasets
dataset_repo = "XavierJiezou/Cloud-Adapter"  # Hugging Face repository name
dataset_names = [
    "hrc_whu",
    "gf12ms_whu_gf1",
    "gf12ms_whu_gf2",
    "cloudsen12_high_l1c",
    "cloudsen12_high_l2a",
    "l8_biome",
]

# Function to create a ZIP file for a dataset directory
def create_zip(dataset_path, output_path):
    """
    Compress a dataset directory into a ZIP file.
    Args:
        dataset_path (str): Path to the dataset directory.
        output_path (str): Path to save the ZIP file.
    """
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(dataset_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, dataset_path)
                zipf.write(file_path, arcname)
    print(f"Compressed {dataset_path} into {output_path}")

# Function to upload ZIP files to Hugging Face Hub
def upload_zip_to_hub(dataset_name, zip_path, repo_name):
    """
    Upload a ZIP file to a Hugging Face repository.
    Args:
        dataset_name (str): Name of the dataset (used as a file identifier).
        zip_path (str): Path to the ZIP file.
        repo_name (str): Hugging Face repository name.
    """
    api = HfApi()
    token = HfFolder.get_token()
    file_name = f"{dataset_name}.zip"
    api.upload_file(
        path_or_fileobj=zip_path,
        path_in_repo=file_name,
        repo_id=repo_name,
        repo_type="dataset",
        token=token,
    )
    print(f"Uploaded {file_name} to {repo_name}")

# Main script
if __name__ == "__main__":
    for dataset_name in dataset_names:
        dataset_path = os.path.join(base_root, dataset_name)
        if not os.path.exists(dataset_path):
            print(f"Dataset directory does not exist: {dataset_path}")
            continue
        
        # Create ZIP file
        zip_path = f"{dataset_name}.zip"
        create_zip(dataset_path, zip_path)
        
        # Upload ZIP file to Hugging Face Hub
        upload_zip_to_hub(dataset_name, zip_path, dataset_repo)
