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
        self.img_labels = pd.read_csv(annotations_file_path)
        self.img_labels = self.img_labels[['id_x', 'label', 'root_category']]
        self.image_directory = image_directory
        self.transform = transform
        self.target_transform = target_transform
        unique_categories = self.img_labels['root_category'].unique()
        self.label_encoder = OrderedDict(
            (category, idx) for idx, category in enumerate(unique_categories))
        self.label_decoder = OrderedDict(
            (idx, category) for category, idx in self.label_encoder.items())

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.target_transform = target_transform

    def __len__(self):
        print("Calculating dataset length...")
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_filename = self.img_labels.iloc[idx, 0]
        img_path = os.path.join(self.image_directory, img_filename + '.jpg')
        if not os.path.isfile(img_path):
            img_path = os.path.join(
                self.image_directory, img_filename + '.png')
            if not os.path.isfile(img_path):
                raise FileNotFoundError(f"Image file not found: {img_path}")
        image = Image.open(img_path)
        image = transforms.Resize((224, 224))(
            image)  # Resize the image to 224x224
        label = self.label_encoder[self.img_labels.iloc[idx, 2]]

        return image, label


annotations_file_path = r"C:\Users\nacho\New folder\AiCore\Facebook_Project\training_data.csv"
image_directory = r"C:\Users\nacho\New folder\AiCore\Facebook_Project\Cleaned_images"

print("Creating dataset...")
dataset = ImageDataset(annotations_file_path, image_directory,
                       transform=None, target_transform=None)
print("Dataset created.")

print("Dataset length:", len(dataset))
sample_image, sample_label = dataset[0]
sample_image = torch.tensor(np.array(sample_image))
print("Sample image shape:", sample_image.shape)
print("Sample label:", sample_label)
print("\nLabel Encoder:")
print(dataset.label_encoder)
print("\nLabel Decoder:")
print(dataset.label_decoder)
