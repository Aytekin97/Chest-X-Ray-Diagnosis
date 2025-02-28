import tarfile
import os

# Path to the folder containing the .tar.gz files
images_dir = "THE GIANT DATASET/CXR8/images"

# Extract all .tar.gz files in the folder
for file in os.listdir(images_dir):
    if file.endswith(".tar.gz"):
        file_path = os.path.join(images_dir, file)
        print(f"Extracting {file_path}...")
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=images_dir)  # Extract files into the same directory
