import os
import json

# Directory containing your images
image_dir = r"C:\\Users\\nacho\\New folder\\AiCore\\Facebook_Project\\Cleaned_images"

# Get list of image filenames
image_filenames = sorted(os.listdir(image_dir))

# Create the mapping dictionary
id_to_filename = {f"embedding_{i}": filename for i,
                  filename in enumerate(image_filenames)}

# Directory where you want to save the JSON file
output_dir = r"C:\\Users\\nacho\\New folder\\AiCore\\Facebook_Project"

# Ensure the output directory exists; create it if necessary
os.makedirs(output_dir, exist_ok=True)

# Save the mapping to a JSON file in the output directory
mapping_path = os.path.join(output_dir, 'id_to_filename.json')
try:
    with open(mapping_path, 'w') as f:
        json.dump(id_to_filename, f)
    print(f"Mapping file created successfully at: {mapping_path}")
except Exception as e:
    print(f"Error occurred while saving JSON file: {e}")
