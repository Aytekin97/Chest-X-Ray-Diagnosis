import pandas as pd
# Split labels into individual conditions
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import random
from PIL import Image
import os
import numpy as np


# Define transformations for data augmentation
augmentation_transforms = transforms.Compose([
    #transforms.RandomHorizontalFlip(p=0.5),                       # Horizontal flipping
    transforms.RandomRotation(degrees=15),                        # Random rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2),          # Random brightness and contrast adjustment
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),     # Random affine transformations
    transforms.RandomResizedCrop(512, scale=(0.9, 1.0)),          # Random resizing and cropping
    #transforms.RandomApply([transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x))], p=0.5),  # Add Gaussian noise
    #transforms.Normalize(mean=[0.5], std=[0.5])                   # Normalize pixel values
])


# Augment images to increase dataset size
def augment_images(data_df, images_dir, target_size=4300):
    """
    Augments images from `data_df` until each class reaches `target_size`.
    Saves augmented images locally and returns a new balanced dataset.
    """
    augmented_data = []
    class_counts = data_df['Finding Labels'].value_counts().to_dict()

    for label, count in class_counts.items():
        if count >= target_size:
            augmented_data.append(data_df[data_df['Finding Labels'] == label])
            continue

        class_data = data_df[data_df['Finding Labels'] == label]
        images_needed = target_size - count
        augmented_images = []
        augmented_labels = []

        while len(augmented_images) < images_needed:
            row = class_data.sample(n=1).iloc[0]
            img_name = row['Image Index']
            img_path = os.path.join(images_dir, img_name)

            if not os.path.exists(img_path):
                continue  # Skip missing images

            # Open image and apply augmentation
            img = Image.open(img_path).convert('RGB')
            img = augmentation_transforms(img)  # Apply transformations
            
            # Save the augmented image with a unique filename
            aug_img_name = f"AUG_{len(augmented_images)}_{img_name}"
            aug_img_path = os.path.join(images_dir, aug_img_name)
            img.save(aug_img_path)  # Save the augmented image
            
            # Add augmented image details to the dataset
            augmented_images.append(aug_img_name)
            augmented_labels.append(label)

        # Create augmented DataFrame
        augmented_df = pd.DataFrame({
            'Image Index': augmented_images,
            'Finding Labels': augmented_labels
        })
        augmented_data.append(augmented_df)

    return pd.concat(augmented_data, ignore_index=True)

def get_data():
    # Load metadata
    metadata = pd.read_csv("THE GIANT DATASET/CXR8/Data_Entry_2017_v2020.csv")

    # Filter relevant classes
    filtered_data = metadata[
        metadata['Finding Labels'].isin(['Effusion', 'Pneumothorax', 'Atelectasis', 'No Finding'])
    ]

    # Define target size for balancing
    target_size = 6000
    IMAGES_DIR = "THE GIANT DATASET/CXR8/images"

    # Get `No Finding` samples without augmentation (already has enough samples)
    no_finding_data = filtered_data[filtered_data['Finding Labels'] == 'No Finding']
    no_finding_sample = no_finding_data.sample(n=target_size, random_state=42)

    # Get minority classes
    atelectasis = filtered_data[filtered_data['Finding Labels'] == 'Atelectasis']
    effusion = filtered_data[filtered_data['Finding Labels'] == 'Effusion']
    pneumothorax = filtered_data[filtered_data['Finding Labels'] == 'Pneumothorax']

    # Apply augmentation only to generate the missing amount
    aug_atelectasis = augment_images(atelectasis, IMAGES_DIR, target_size)
    aug_effusion = augment_images(effusion, IMAGES_DIR, target_size)
    aug_pneumothorax = augment_images(pneumothorax, IMAGES_DIR, target_size)
    #aug_no_finding = augment_images(no_finding_sample, IMAGES_DIR, target_size)

    # Combine the original data with the augmented data
    balanced_atelectasis = pd.concat([atelectasis, aug_atelectasis])
    balanced_effusion = pd.concat([effusion, aug_effusion])
    balanced_pneumothorax = pd.concat([pneumothorax, aug_pneumothorax])
    #balanced_no_finding = pd.concat([no_finding_sample, aug_no_finding])

    # Combine all classes
    balanced_data = pd.concat([no_finding_sample, balanced_atelectasis, balanced_effusion, balanced_pneumothorax])

    # Shuffle dataset
    balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Stratified train-test split
    train_data, temp_data = train_test_split(
        balanced_data, test_size=0.30, stratify=balanced_data['Finding Labels'], random_state=42
    )
    val_data, test_data = train_test_split(
        temp_data, test_size=0.50, stratify=temp_data['Finding Labels'], random_state=42
    )

    return train_data, test_data, val_data


def preprocess_image(image_path, target_size=(512, 512)):
    """
    Resizes a 1024x1024 image to 512x512 and normalizes pixel values.
    
    Args:
        image_path (str): Path to the image file.
        target_size (tuple): Desired output size (default: 512x512).
    
    Returns:
        np.array: Preprocessed image as a NumPy array.
    """
    # Open image and convert to RGB
    img = Image.open(image_path).convert('RGB')
    
    # Resize directly (no need to maintain aspect ratio or pad)
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    # Normalize pixel values to [0, 1]
    img_array = np.array(img) / 255.0

    return img_array


def show_sample_images(data_df, images_dir, num_samples=5):
    """
    Displays a few random images from the dataset along with their labels.

    Args:
        data_df (pd.DataFrame): Dataframe containing image metadata.
        images_dir (str): Path to the directory with images.
        num_samples (int): Number of images to display.
    """
    sample_data = data_df.sample(n=num_samples, random_state=42)  # Randomly select images

    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))

    for i, (_, row) in enumerate(sample_data.iterrows()):
        img_path = os.path.join(images_dir, row["Image Index"])
        img = Image.open(img_path).convert("RGB")  # Open image

        axes[i].imshow(img)
        axes[i].set_title(row["Finding Labels"])  # Show label as title
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()
