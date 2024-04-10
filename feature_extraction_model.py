import torch
import torch.nn as nn
from torchvision import models, transforms
import pickle
from Image_Dataset import ImageDataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import numpy as np

data_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(-20, 20)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.2),
    transforms.RandomCrop(size=(224, 224), padding=16),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

annotations_file_path = r"C:\Users\nacho\New folder\AiCore\Facebook_Project\training_data.csv"
image_directory = r"C:\Users\nacho\New folder\AiCore\Facebook_Project\Cleaned_images"
dataset = ImageDataset(annotations_file_path,
                       image_directory, transform=data_augmentation)


weights_folder = r"C:\Users\nacho\New folder\AiCore\Facebook_Project\model_evaluation\weights"


def custom_collate_fn(batch):
    images, labels = zip(*batch)
    images = [data_augmentation(image) for image in images]
    images = torch.stack(images, dim=0)
    labels = torch.LongTensor(labels)
    return images, labels


train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size])

batch_size = 32
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)


writer = SummaryWriter(
    r"C:\Users\nacho\New folder\AiCore\Facebook_Project\SummaryWriter")


resnet50 = models.resnet50(pretrained=True)
feature_model = nn.Sequential(*list(resnet50.children())[:-1])
num_features = resnet50.fc.in_features
feature_model.add_module('fc', nn.Linear(num_features, 1000))

for param in feature_model.parameters():
    param.requires_grad = False

all_features = []
all_labels = []

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet50.fc.parameters(), lr=0.002)

for inputs, labels in train_loader:
    features = feature_model(inputs).detach().cpu().numpy()
    all_features.append(features)
    all_labels.append(labels)

all_features = np.concatenate(all_features, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

final_model_folder = r"C:\Users\nacho\New folder\AiCore\Facebook_Project\final_model"
os.makedirs(final_model_folder, exist_ok=True)
torch.save(feature_model.state_dict(), os.path.join(
    final_model_folder, 'image_model.pt'))

num_epochs = 10

for epoch in range(num_epochs):
    resnet50.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()

        outputs = resnet50(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

    resnet50.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = resnet50(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            val_accuracy += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_accuracy /= len(val_dataset)

    writer.add_scalar("Loss/Validation", val_loss, epoch)
    writer.add_scalar('Accuracy/validation', val_accuracy, epoch)

    weights_filename = f"weights_epoch_{epoch+1}.pth"
    weights_path = os.path.join(weights_folder, weights_filename)
    torch.save(resnet50.state_dict(), weights_path)

    print(
        f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')

current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
folder_name = f"weights_{current_datetime}"
path = os.path.join(
    r"C:\Users\nacho\New folder\AiCore\Facebook_Project\model_evaluation", folder_name)
os.makedirs(path, exist_ok=True)
weights_path = os.path.join(path, "final_weights.pth")
torch.save(resnet50.state_dict(), weights_path)


with open('image_decoder.pkl', 'wb') as f:
    pickle.dump(dataset.label_decoder, f)
