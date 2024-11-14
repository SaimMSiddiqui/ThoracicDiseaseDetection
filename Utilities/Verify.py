import csv
import os

# Define paths
csv_file = 'path_to_your_csv.csv'  # Update with the path to your CSV file
folder_to_check = 'D:\\Labeled'  # Folder where you expect the images to be

# Step 1: Extract unique image names from the CSV
image_set = set()  # Use a set to handle duplicates automatically

with open(csv_file, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        image_name = row['Image ID']
        image_set.add(image_name)

# Step 2: Get a list of files in the folder
files_in_folder = set(os.listdir(folder_to_check))

# Step 3: Verify each image in the CSV is in the folder
missing_files = image_set - files_in_folder  # Files in CSV but not in the folder
extra_files = files_in_folder - image_set    # Files in folder but not in CSV

# Step 4: Output results
if missing_files:
    print("The following images from the CSV are missing in the folder:")
    for file in missing_files:
        print(file)
else:
    print("All images from the CSV are present in the folder.")

if extra_files:
    print("\nThe following files are in the folder but not listed in the CSV:")
    for file in extra_files:
        print(file)
else:
    print("\nNo extra files in the folder.")

print("Verification complete.")
