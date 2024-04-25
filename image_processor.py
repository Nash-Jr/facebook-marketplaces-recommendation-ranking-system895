import torch
import torch.nn as nn
from torchvision import models, transforms
import pickle
from Image_Dataset import ImageDataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

from feature_extraction_model import feature_model, data_augmentation, custom_collate_fn

data_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(-20, 20)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.2),
    transforms.RandomCrop(size=(224, 224), padding=16),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def custom_collate_fn(batch):
    images, labels = zip(*batch)
    images = [data_augmentation(image) for image in images]
    images = torch.stack(images, dim=0)
    labels = torch.LongTensor(labels)
    return images, labels


annotations_file_path = r"C:\Users\nacho\New folder\AiCore\Facebook_Project\training_data.csv"
image_directory = r"C:\Users\nacho\New folder\AiCore\Facebook_Project\Cleaned_images"
dataset = ImageDataset(annotations_file_path,
                       image_directory, transform=data_augmentation)

resnet50 = models.resnet50(pretrained=True)
num_features = resnet50.fc.in_features
num_categories = len(dataset.label_decoder)
resnet50.fc = nn.Linear(num_features, num_categories)
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_categories)
model.load_state_dict(torch.load(
    r"C:\Users\nacho\New folder\AiCore\Facebook_Project\model_evaluation\weights_20240405_013955"))
model.eval()


with open('image_decoder.pkl', 'rb') as f:
    label_decoder = pickle.load(f)

new_image_path = r"C:\Users\nacho\New folder\AiCore\Facebook_Project\Cleaned_images\new_image.jpg"
new_dataset = ImageDataset([new_image_path], [0],
                           label_decoder, transform=data_augmentation)
new_dataloader = torch.utils.data.DataLoader(
    new_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

with torch.no_grad():
    for inputs, labels in new_dataloader:
        features = feature_model(inputs).detach().cpu().numpy()
