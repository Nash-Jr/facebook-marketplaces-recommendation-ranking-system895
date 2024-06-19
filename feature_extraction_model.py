import torch
import torch.nn as nn
from torchvision import models, transforms
import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from Image_Dataset import ImageDataset
from PIL import Image
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import timm


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(
                    f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def custom_collate_fn(batch, data_augmentation, normalize_transform):
    images, labels, image_ids = zip(*batch)
    augmented_images = [data_augmentation(image) for image in images]
    augmented_tensors = [transforms.Resize((224, 224))(
        transforms.ToTensor()(image)) for image in augmented_images]
    augmented_tensors = torch.stack(augmented_tensors, dim=0)
    augmented_tensors = normalize_transform(augmented_tensors)
    labels = torch.LongTensor([label for label in labels])
    return augmented_tensors, labels, image_ids


def show_images(images, labels, label_decoder):
    fig, axes = plt.subplots(1, len(images), figsize=(12, 12))
    if len(images) == 1:
        axes = [axes]
    for img, label, ax in zip(images, labels, axes):
        ax.imshow(transforms.ToPILImage()(img))
        ax.set_title(label_decoder.inverse_transform([label])[0])
        ax.axis('off')
    plt.show()


def main(args):
    data_augmentation = timm.data.auto_augment.rand_augment_transform(
        "rand-m9-mstd0.5-inc1",
        {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)}
    )

    normalize_transform = transforms.Normalize(
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    dataset = ImageDataset(args.annotations_file,
                           args.image_directory, transform=data_augmentation)

    for i in range(5):
        image, label, image_id = dataset[i]
        print(
            f"Image: {dataset.image_ids[i]}, Label: {list(dataset.label_encoder.keys())[list(dataset.label_encoder.values()).index(label)]}")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])

    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=lambda batch: custom_collate_fn(batch, data_augmentation, normalize_transform))

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                             collate_fn=lambda batch: custom_collate_fn(batch, data_augmentation, normalize_transform))

    writer = SummaryWriter(args.summary_writer_path)

    resnet50 = models.resnet50(pretrained=True)
    for param in resnet50.parameters():
        param.requires_grad = False
    feature_model = nn.Sequential(*list(resnet50.children())[:-1])
    num_features = resnet50.fc.in_features

    for layer in resnet50.children():
        if isinstance(layer, nn.Conv2d):
            for param in layer.parameters():
                param.requires_grad = False
        elif isinstance(layer, nn.Linear):
            for param in layer.parameters():
                param.requires_grad = True

    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output_size = feature_model(dummy_input).flatten().shape[0]

    final_layer = nn.Linear(num_features, 1024)
    final_layer = nn.Sequential(
        final_layer, nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, 13))
    resnet50.fc = final_layer

    for layer in resnet50.layer4.children():
        for param in layer.parameters():
            param.requires_grad = True

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet50.parameters(), lr=0.002)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    num_epochs = 50

    early_stopping = EarlyStopping(patience=10, verbose=True)

    for epoch in range(num_epochs):
        resnet50.train()
        for inputs, labels, _ in train_loader:
            optimizer.zero_grad()
            outputs = resnet50(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        resnet50.eval()
        val_loss = 0
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                outputs = resnet50(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        val_loss /= len(val_loader)
        val_accuracy = accuracy_score(all_labels, all_predictions)
        val_precision = precision_score(
            all_labels, all_predictions, average='macro')
        val_recall = recall_score(all_labels, all_predictions, average='macro')
        val_f1 = f1_score(all_labels, all_predictions, average='macro')

        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar('Accuracy/validation', val_accuracy, epoch)
        writer.add_scalar('Precision/validation', val_precision, epoch)
        writer.add_scalar('Recall/validation', val_recall, epoch)
        writer.add_scalar('F1-Score/validation', val_f1, epoch)

        print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}, Validation Precision: {val_precision}, Validation Recall: {val_recall}, Validation F1-Score: {val_f1}')

        scheduler.step()
        early_stopping(val_loss, resnet50)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        weights_folder = args.weights_folder
        os.makedirs(weights_folder, exist_ok=True)
        weights_filename = f"weights_epoch_{epoch + 1}.pth"
        weights_path = os.path.join(weights_folder, weights_filename)
        torch.save(resnet50.state_dict(), weights_path)

    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_weights_path = os.path.join(
        args.model_evaluation_path, f"final_weights_{current_datetime}.pth")
    os.makedirs(args.model_evaluation_path)
    torch.save(resnet50.state_dict(), final_weights_path)

    final_model_folder = args.final_model_path
    os.makedirs(final_model_folder, exist_ok=True)
    final_model_path = os.path.join(final_model_folder, 'image_model.pt')
    if os.path.exists(final_model_path):
        os.remove(final_model_path)
    torch.save(resnet50.state_dict(), final_model_path)


if __name__ == "__main__":
    class Args:
        annotations_file = "C:\\Users\\nacho\\New folder\\AiCore\\Facebook_Project\\training_data.csv"
        image_directory = "C:\\Users\\nacho\\New folder\\AiCore\\Facebook_Project\\Cleaned_images"
        summary_writer_path = "C:\\Logs\\TensorBoard"
        weights_folder = "C:\\Users\\nacho\\New folder\\AiCore\\Facebook_Project\\model_evaluation\\weights"
        model_evaluation_path = "C:\\Users\\nacho\\New folder\\AiCore\\Facebook_Project\\model_evaluation\\weights_20240405_013955"
        final_model_path = "C:\\Users\\nacho\\New folder\\AiCore\\Facebook_Project\\final_model"

    args = Args()

    main(args)
