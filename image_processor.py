import torch
import torch.nn as nn
from torchvision import models, transforms
import pickle
from Image_Dataset import ImageDataset
from PIL import Image
import numpy as np

from feature_extraction_model import feature_model

data_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(-20, 20)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.2),
    transforms.RandomCrop(size=(224, 224), padding=16)
])

normalize_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def custom_collate_fn(batch):
    images, labels = zip(*batch)
    pil_images = []
    for image in images:
        image_np = np.asarray(image)
        image_pil = Image.fromarray(image_np.astype(np.uint8))
        pil_images.append(image_pil)

    augmented_images = [data_augmentation(image) for image in pil_images]
    augmented_tensors = [normalize_transform(
        image) for image in augmented_images]
    augmented_tensors = torch.stack(augmented_tensors, dim=0)
    labels = torch.LongTensor(labels)
    return augmented_tensors, labels


annotations_file_path = r"C:\Users\nacho\New folder\AiCore\Facebook_Project\training_data.csv"
image_directory = r"C:\Users\nacho\New folder\AiCore\Facebook_Project\Cleaned_images"
dataset = ImageDataset(annotations_file_path, image_directory, transform=None)

resnet50 = models.resnet50(pretrained=True)
num_features = resnet50.fc.in_features
num_categories = len(dataset.label_decoder)
resnet50.fc = nn.Linear(num_features, num_categories)
model = resnet50
model.load_state_dict(torch.load(
    r"C:\Users\nacho\New folder\AiCore\Facebook_Project\model_evaluation\weights_20240405_013955\final_weights.pth"))
model.eval()

with open('image_decoder.pkl', 'rb') as f:
    label_decoder = pickle.load(f)

new_image_path = r"C:\Users\nacho\New folder\AiCore\Facebook_Project\Cleaned_images\new_image.jpg"
new_dataset = ImageDataset(
    [new_image_path], [label_decoder['your_class_name']], transform=None)
new_dataloader = torch.utils.data.DataLoader(
    new_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

with torch.no_grad():
    for inputs, labels in new_dataloader:
        features = feature_model(inputs)
        print("Features shape:", features.shape)
