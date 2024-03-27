import torch
import torch.nn as nn
from torchvision import models, transforms
import pickle
from Image_Dataset import ImageDataset
from torch.utils.tensorboard import SummaryWriter


annotations_file_path = r"C:\Users\nacho\New folder\AiCore\Facebook_Project\training_data.csv"
image_directory = r"C:\Users\nacho\New folder\AiCore\Facebook_Project\Cleaned_images"
dataset = ImageDataset(annotations_file_path, image_directory, )


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def custom_collate_fn(batch):
    images, labels = zip(*batch)
    images = [transform(image) for image in images]
    images = torch.stack(images, dim=0)
    labels = torch.LongTensor(labels)
    return images, labels


train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size])

batch_size = 64
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)


writer = SummaryWriter(r"C:\Users\nacho\New folder\AiCore\Facebook_Project")

resnet50 = models.resnet50(pretrained=True)

for param in resnet50.parameters():
    param.requires_grad = False

num_features = resnet50.fc.in_features
num_categories = len(dataset.label_decoder)
resnet50.fc = nn.Linear(num_features, num_categories)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet50.fc.parameters(), lr=0.001)

num_epochs = 15

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

    print(
        f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')

with open('image_decoder.pkl', 'wb') as f:
    pickle.dump(dataset.label_decoder, f)
