import csv
import os
import shutil

# Define paths
csv_file = 'path_to_your_csv.csv'  # Update with the path to your CSV file
decompressed_folder = 'D:\\data dump'  # Update with the path to your decompressed folders
destination_folder = 'D:\\Labeled'  # Folder to move the labeled images

# Create destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Step 1: Extract unique image names from the CSV
image_set = set()  # Use a set to automatically handle duplicates

with open(csv_file, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        image_name = row['Image ID']
        image_set.add(image_name)

# Step 2: Search for images in decompressed folders and move to destination
for root, dirs, files in os.walk(decompressed_folder):
    for file in files:
        if file in image_set:  # Check if the file is in the list from the CSV
            source_path = os.path.join(root, file)
            destination_path = os.path.join(destination_folder, file)
            # Move the file
            shutil.move(source_path, destination_path)
            print(f'Moved {file} to {destination_folder}')

print("All labeled images have been moved to the 'Labeled' folder.")
