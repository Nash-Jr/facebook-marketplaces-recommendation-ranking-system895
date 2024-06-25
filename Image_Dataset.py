import os
import pandas as pd
from torchvision.io import read_image
import torch
from collections import OrderedDict
from torchvision import transforms
from PIL import Image
import numpy as np


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file_path, image_directory, transform=None, target_transform=None):
        print("Initializing dataset...")
        self.annotations_file_path = annotations_file_path
        self.image_directory = image_directory
        self.transform = transform
        self.target_transform = target_transform

        self.annotations = pd.read_csv(self.annotations_file_path)

        self.image_ids = self.annotations['id_x'].tolist()
        self.annotations = self.annotations[['id_x', 'label', 'root_category']]

        unique_categories = self.annotations['label'].unique()
        self.label_encoder = OrderedDict(
            (category, idx) for idx, category in enumerate(unique_categories))
        self.label_decoder = OrderedDict(
            (idx, category) for category, idx in self.label_encoder.items())

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_filename = self.annotations.loc[idx, 'id_x']
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.image_directory, img_filename + '.jpg')
        if not os.path.isfile(img_path):
            img_path = os.path.join(
                self.image_directory, img_filename + '.png')
            if not os.path.isfile(img_path):
                raise FileNotFoundError(f"Image file not found: {img_path}")

        image = Image.open(img_path).convert('RGB')
        label = self.label_encoder[self.annotations.loc[idx, 'label']]

        if self.transform:
            image = self.transform(image)

        return image, label, image_id


annotations_file_path = r"C:\Users\nacho\New folder\AiCore\Facebook_Project\training_data.csv"
image_directory = r"C:\Users\nacho\New folder\AiCore\Facebook_Project\Cleaned_images"
save_path = r"C:\Users\nacho\New folder\AiCore\Facebook_Project\saved_dataset.pth"

print("Creating dataset...")
dataset = ImageDataset(annotations_file_path, image_directory,
                       transform=None, target_transform=None)
print("Dataset created.")

print("Dataset length:", len(dataset))
sample_image, sample_label, sample_image_id = dataset[0]
sample_image = torch.tensor(np.array(sample_image))
print("Sample image shape:", sample_image.shape)
print("Sample label:", sample_label)
print("\nLabel Encoder:")
print(dataset.label_encoder)
print("\nLabel Decoder:")
print(dataset.label_decoder)

# Save the dataset
torch.save(dataset, save_path)
print(f"Dataset saved to {save_path}")
