import torch
import torch.nn as nn
from torchvision import models, transforms
from Image_Dataset import ImageDataset
import json
import os
import timm
import traceback
from typing import Tuple


def extract_and_save_embeddings(annotations_file, image_directory, final_model_path):
    print("Initializing dataset...")
    # Initialize the dataset with the same data augmentation transform
    data_augmentation = timm.data.auto_augment.rand_augment_transform(
        "rand-m9-mstd0.5-inc1",
        {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)}
    )
    dataset = ImageDataset(
        annotations_file, image_directory, transform=data_augmentation)

    print("Loading pre-trained model...")
    # Load the custom-trained model
    resnet50 = models.resnet50(pretrained=True)
    for param in resnet50.parameters():
        param.requires_grad = False
    num_features = resnet50.fc.in_features
    final_layer = nn.Linear(num_features, 1024)
    final_layer = nn.Sequential(
        final_layer, nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, 13))
    resnet50.fc = final_layer
    model_dict = resnet50.state_dict()
    # Load the pre-trained state_dict
    state_dict = torch.load(final_model_path)
    model_dict = resnet50.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    resnet50.load_state_dict(model_dict)

    print("Creating feature extraction model...")
    feature_model = nn.Sequential(*list(resnet50.children())[:-1])
    feature_model.eval()
    # Dictionary to store image embeddings
    image_embeddings = {}

    # Transformation to convert PIL images to tensors
    to_tensor = transforms.ToTensor()
    normalize_transform = transforms.Normalize(
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def process_image(image, label, image_id) -> Tuple[torch.Tensor, int, str]:
        try:
            # Apply data augmentation and normalization
            image = normalize_transform(to_tensor(image))
            image = image.unsqueeze(0)
            # Extract features using the feature model
            embedding = feature_model(image).squeeze().numpy()
            return embedding, label, image_id
        except Exception as e:
            print(f"Error processing image {image_id}: {e}")
            print(traceback.format_exc())
            return None, None, image_id

    print("Extracting embeddings...")
    with open('image_embeddings.json', 'w') as f:
        f.write('{}')

    with torch.no_grad():
        for i, (image, label, image_id) in enumerate(dataset):
            embedding, label, image_id = process_image(image, label, image_id)
            if embedding is not None:
                # Store the embedding in the dictionary
                image_embeddings[image_id] = embedding.tolist()

                # Save the current embedding to the JSON file
                with open('image_embeddings.json', 'r+') as f:
                    data = json.load(f)
                    data.update({image_id: embedding.tolist()})
                    f.seek(0)
                    json.dump(data, f, indent=4)

            if i % 100 == 0:
                print(f"Processed {i+1} images...")

    print("Embeddings saved successfully!")


if __name__ == "__main__":
    annotations_file = r"C:\Users\nacho\New folder\AiCore\Facebook_Project\training_data.csv"
    image_directory = r"C:\Users\nacho\New folder\AiCore\Facebook_Project\Cleaned_images"
    final_model_path = r"C:\Users\nacho\New folder\AiCore\Facebook_Project\final_model\image_model.pt"

    print("Starting embedding extraction process...")
    extract_and_save_embeddings(
        annotations_file, image_directory, final_model_path)
