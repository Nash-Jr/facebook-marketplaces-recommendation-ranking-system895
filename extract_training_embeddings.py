import os
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
import json
from tqdm import tqdm


def create_feature_model():
    resnet50 = models.resnet50(pretrained=False)
    num_features = resnet50.fc.in_features
    resnet50.fc = nn.Sequential(
        nn.Linear(num_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(1024, 2048)
    )
    return resnet50


def load_model(model_path):
    model = create_feature_model()
    try:
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        print("Successfully loaded model weights")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using model with randomly initialized weights.")
    return model


def extract_features(model, image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model(image)
    return features.cpu().numpy().flatten()


def process_images_batch(model, image_dir, batch_size=32, output_file='embeddings.npz'):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    image_files = [f for f in os.listdir(
        image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    total_batches = (len(image_files) + batch_size - 1) // batch_size

    embeddings = []
    filenames = []

    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    for i in tqdm(range(0, len(image_files), batch_size), total=total_batches, desc="Processing batches"):
        batch_files = image_files[i:i+batch_size]
        batch_embeddings = []

        for img_file in batch_files:
            img_path = os.path.join(image_dir, img_file)
            embedding = extract_features(model, img_path, transform)
            batch_embeddings.append(embedding)
            filenames.append(img_file)

        embeddings.extend(batch_embeddings)

        # Save intermediate results every 10 batches
        if (i // batch_size + 1) % 10 == 0:
            np.savez_compressed(
                output_file,
                embeddings=np.array(embeddings),
                filenames=np.array(filenames)
            )
            print(
                f"Saved intermediate results. Processed {len(embeddings)} images so far.")

    # Save final results
    np.savez_compressed(
        output_file,
        embeddings=np.array(embeddings),
        filenames=np.array(filenames)
    )
    print(f"Finished processing. Total images processed: {len(embeddings)}")


current_directory = os.path.dirname(os.path.abspath(__file__))
model_filename = "checkpoint.pt"
model_path = os.path.join(current_directory, model_filename)

# Load the model
feature_model = load_model(model_path)

# Set image directory and process images
image_dir = r"C:\Users\nacho\New folder\AiCore\Facebook_Project\Cleaned_images"
process_images_batch(feature_model, image_dir,
                     batch_size=32, output_file='embeddings.npz')
