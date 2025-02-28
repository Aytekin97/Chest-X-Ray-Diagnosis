import pandas as pd
# Split labels into individual conditions
from collections import Counter
from sklearn.model_selection import train_test_split
import numpy as np
from prepare_data import get_data, show_sample_images

# Load metadata
metadata = pd.read_csv("THE GIANT DATASET/CXR8/Data_Entry_2017_v2020.csv")

# Check basic information
print(metadata.info())
print(metadata.head())
all_labels = metadata['Finding Labels'].str.split('|').explode()
label_counts = Counter(all_labels)
print(label_counts)

filtered_data = metadata[
    metadata['Finding Labels'].isin(['Effusion', 'Pneumothorax', 'Atelectasis'])
]

# Calculate the mean count of the three target labels
mean_sample_size = int(filtered_data['Finding Labels'].value_counts().mean())
print(f"Mean sample size for target conditions: {mean_sample_size}")

no_finding_data = metadata[metadata['Finding Labels'] == 'No Finding']
no_finding_sample = no_finding_data.sample(n=4300, random_state=42)

final_data = pd.concat([filtered_data, no_finding_sample], ignore_index=True)

print(final_data['Finding Labels'].value_counts())


# Stratified split
train_data, temp_data = train_test_split(
    final_data, test_size=0.30, stratify=final_data['Finding Labels'], random_state=42
)
val_data, test_data = train_test_split(
    temp_data, test_size=0.50, stratify=temp_data['Finding Labels'], random_state=42
)

# Check splits
print(f"\nTrain data shape: {train_data.shape}")
print("\nTraining set class distribution:")
print(train_data['Finding Labels'].value_counts())

print("\nValidation set class distribution:")
print(val_data['Finding Labels'].value_counts())

print("\nTest set class distribution:")
print(test_data['Finding Labels'].value_counts())

train_data, test_data, val_data = get_data()

print("\nget_data() Training set class distribution:")
print(train_data['Finding Labels'].value_counts())

print("\nget_data() Validation set class distribution:")
print(val_data['Finding Labels'].value_counts())

print("\nget_data() Test set class distribution:")
print(test_data['Finding Labels'].value_counts())


# Path to images
IMAGES_DIR = "THE GIANT DATASET/CXR8/images"

# Show random images from training data
show_sample_images(train_data, IMAGES_DIR, num_samples=10)