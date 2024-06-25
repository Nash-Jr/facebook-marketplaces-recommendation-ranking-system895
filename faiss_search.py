import os
import torch
import torch.nn as nn
from torchvision import models, transforms
import faiss
import numpy as np
from PIL import Image
import json
from matplotlib import pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def create_feature_model():
    resnet50 = models.resnet50(pretrained=False)
    num_features = resnet50.fc.in_features
    resnet50.fc = nn.Sequential(
        nn.Linear(num_features, 1024),  # Hidden layer
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(1024, 2048)  # Output layer matching FAISS index dimension
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


# Set up paths
current_directory = os.path.dirname(os.path.abspath(__file__))
model_filename = "checkpoint.pt"
model_path = os.path.join(current_directory, model_filename)
index_path = os.path.join(current_directory, "index.faiss")

# Load the model
feature_model = load_model(model_path)
feature_model.eval()

# Print model summary
print("\nModel Summary:")
print(feature_model)

# Print number of trainable parameters
total_params = sum(p.numel()
                   for p in feature_model.parameters() if p.requires_grad)
print(f"\nTotal trainable parameters: {total_params}")


def extract_features(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        features = feature_model(image)

    return features.cpu().numpy().flatten()


def load_large_embedding_arrays(file_path):
    embeddings = []
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # Fixing the structure by wrapping in a list and splitting objects correctly
        fixed_content = "[" + content.replace("}{", "},{") + "]"

        # Load the entire content as a JSON object
        data = json.loads(fixed_content)

        # Iterate over each key, value pair in the JSON objects
        for item in data:
            for key, value in item.items():
                if isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):
                    embeddings.append(np.array(value))
                else:
                    print(f"Skipping invalid embedding for key {key}: {value}")

    except (ValueError, json.JSONDecodeError) as e:
        print(f"Error loading JSON: {e}")
        print(f"Problematic content: {content[:500]}...")

    return embeddings


def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


# Load the embeddings file
embeddings_path = os.path.join(current_directory, 'image_embeddings.json')
embeddings = load_large_embedding_arrays(embeddings_path)

if not embeddings:
    print("Failed to load image embeddings. Please check the file.")
    exit(1)

# Prepare data for FAISS
image_ids = [f"embedding_{i}" for i in range(len(embeddings))]
embeddings_array = np.array(embeddings).astype('float32')

embeddings_array = normalize_embeddings(embeddings_array)

# Create FAISS index
dimension = embeddings_array.shape[1]  # Dimensionality of the embeddings
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_array)

faiss.write_index(index, index_path)
print(f"FAISS index saved to {index_path}")


def search_similar_images(query_image_path, top_k=5):
    query_features = extract_features(query_image_path)

    query_features = normalize_embeddings(query_features.reshape(1, -1))

    # Debugging: Print dimensions of query features and FAISS index
    print(f"Query features shape: {query_features.shape}")
    print(f"FAISS index dimension: {index.d}")

    distances, indices = index.search(query_features, top_k)

    similar_images = [image_ids[i] for i in indices[0]]
    return similar_images, distances[0]


# Example usage
query_image_path = r"C:\Users\nacho\New folder\AiCore\Facebook_Project\Cleaned_images\00b75a5a-8270-440b-90da-cd2841368acf.jpg"
similar_images, distances = search_similar_images(query_image_path)

print("Similar images:")
for image_id, distance in zip(similar_images, distances):
    print(f"Image ID: {image_id}, Distance: {distance}")

# Load mapping of embedding IDs to filenames
mapping_path = os.path.join(current_directory, 'id_to_filename.json')
with open(mapping_path, 'r') as f:
    id_to_filename = json.load(f)


def display_images(image_paths, titles=None):
    fig, axes = plt.subplots(1, len(image_paths), figsize=(15, 5))
    for i, image_path in enumerate(image_paths):
        image = Image.open(image_path)
        axes[i].imshow(image)
        if titles:
            axes[i].set_title(titles[i])
        axes[i].axis('off')
    plt.show()


# Visualise results
image_dir = r"C:\\Users\\nacho\\New folder\\AiCore\\Facebook_Project\\Cleaned_images"
similar_image_paths = [os.path.join(
    image_dir, id_to_filename[img_id]) for img_id in similar_images]
display_images([query_image_path] + similar_image_paths, ["Query"] +
               [f"Similar {i+1}" for i in range(len(similar_images))])
