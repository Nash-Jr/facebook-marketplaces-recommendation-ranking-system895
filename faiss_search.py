import json
import numpy as np
import faiss
from feature_extraction_model import feature_model
import torch
import os
from clean_images import check_image_channels, resize_image
from PIL import Image, UnidentifiedImageError
from torchvision import transforms


with open('image_embeddings.json', 'r') as f:
    image_embeddings = json.load(f)

embeddings = np.array([emb for emb in image_embeddings.values()])

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)


def clean_and_preprocess_image(image_path, final_size=224):
    try:
        with Image.open(image_path) as img:
            img = check_image_channels(img)
            img = resize_image(final_size, img)
            image_tensor = transforms.ToTensor()(img)
            image_tensor = transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image_tensor)
            return image_tensor
    except (IOError, OSError, ValueError, UnidentifiedImageError):
        print(f"Skipping non-image file: {image_path}")
        return None


def extract_image_features(image_path):
    image = clean_and_preprocess_image(image_path)
    if image is not None:
        with torch.no_grad():
            features = feature_model(image.unsqueeze(0)).detach().cpu().numpy()
        return features
    else:
        return None


def search_similar_images(query_image_directory, top_k=5):
    similar_image_ids = []
    for filename in os.listdir(query_image_directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            query_image_path = os.path.join(query_image_directory, filename)
            query_embedding = extract_image_features(query_image_path)
            if query_embedding is not None:
                distances, indices = index.search(
                    query_embedding.reshape(1, -1), top_k)
                similar_ids = [list(image_embeddings.keys())[idx]
                               for idx in indices[0]]
                similar_image_ids.append((filename, similar_ids))
    return similar_image_ids


query_image_directory = r"C:\Users\nacho\New folder\AiCore\Facebook_Project\Cleaned_images"
similar_image_ids = search_similar_images(query_image_directory, top_k=5)
for query_image, similar_ids in similar_image_ids:
    print(f"Query image: {query_image}")
    print(f"Similar image IDs: {similar_ids}")
    print()
