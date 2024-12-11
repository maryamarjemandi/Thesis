import os
import shutil
import random
import pandas as pd

# Define paths
train_dir = "/l/users/maryam.arjemandi/RecentCornicheFetalUS/LabeledImages/train/"
test_dir = "/l/users/maryam.arjemandi/RecentCornicheFetalUS/LabeledImages/test/"
combined_train_dir = "/l/users/maryam.arjemandi/New_train_flat/"
reduced_test_dir = "/l/users/maryam.arjemandi/New_test_flat/"
os.makedirs(combined_train_dir, exist_ok=True)
os.makedirs(reduced_test_dir, exist_ok=True)

# List of valid image extensions
valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

# Initialize metadata storage
metadata = []

# Calculate remaining images needed for 1000 test images
total_test_images_needed = 1000
test_images_count = {class_folder: 0 for class_folder in os.listdir(test_dir)}

print("Starting image processing...")

# Go through each class in the training and testing folders
for class_folder in os.listdir(train_dir):
    train_class_path = os.path.join(train_dir, class_folder)
    test_class_path = os.path.join(test_dir, class_folder)

    # Check if subfolder exists in both train and test directories
    if not os.path.isdir(train_class_path) or not os.path.isdir(test_class_path):
        print(f"Skipping {class_folder}, not found in both train and test directories.")
        continue

    print(f"Processing class: {class_folder}")
    
    # Get all training and testing images for the class, filtering only valid images
    train_images = [img for img in os.listdir(train_class_path) if os.path.splitext(img)[1].lower() in valid_extensions]
    test_images = [img for img in os.listdir(test_class_path) if os.path.splitext(img)[1].lower() in valid_extensions]

    # Copy all training images to the combined training folder
    for image in train_images:
        src_path = os.path.join(train_class_path, image)
        dest_path = os.path.join(combined_train_dir, image)
        shutil.copy(src_path, dest_path)
        metadata.append([image, class_folder, "Training"])
    print(f"Copied {len(train_images)} training images for class {class_folder}")

    # Determine how many images to copy to the test set for this class
    max_images_for_class = min(len(test_images), total_test_images_needed // len(test_images_count))
    test_images_count[class_folder] = max_images_for_class
    total_test_images_needed -= max_images_for_class

    # Shuffle and copy the selected number of images to reduced test folder, others to combined train folder
    random.shuffle(test_images)
    for image in test_images[:max_images_for_class]:
        src_path = os.path.join(test_class_path, image)
        dest_path = os.path.join(reduced_test_dir, image)
        shutil.copy(src_path, dest_path)
        metadata.append([image, class_folder, "Testing"])
    print(f"Copied {max_images_for_class} testing images for class {class_folder}")

    # Copy remaining test images to combined train folder
    for image in test_images[max_images_for_class:]:
        src_path = os.path.join(test_class_path, image)
        dest_path = os.path.join(combined_train_dir, image)
        shutil.copy(src_path, dest_path)
        metadata.append([image, class_folder, "Training"])
    print(f"Moved remaining {len(test_images) - max_images_for_class} test images to training for class {class_folder}")

print("All images processed. Saving metadata...")

# Save metadata to Excel
metadata_df = pd.DataFrame(metadata, columns=["Image_Name", "Original_Class", "New_Set"])
metadata_df.to_excel("/path/to/image_metadata.xlsx", index=False)

print("Images combined successfully, and metadata saved.")
