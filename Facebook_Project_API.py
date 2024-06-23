import io
import json
import pickle
import numpy as np
from PIL import Image
from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import logging
from torchvision import models, transforms
import torch.nn as nn
import torch
import faiss
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
if sys.platform.startswith('win'):
    try:
        import ctypes
        ctypes.CDLL('libiomp5md.dll')
    except OSError:
        pass

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

current_directory = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_directory, "checkpoint.pt")
decoder_path = os.path.join(current_directory, "image_decoder.pkl")
index_path = os.path.join(current_directory, "index.faiss")
id_to_filename_path = os.path.join(current_directory, "id_to_filename.json")

with open(decoder_path, "rb") as f:
    decoder = pickle.load(f)
logger.debug(f"Sample of decoder contents: {list(decoder.items())[:5]}")
logger.debug(f"Number of entries in decoder: {len(decoder)}")

index = faiss.read_index(index_path)
logger.debug(f"Loaded FAISS index with dimension: {index.d}")
logger.debug(f"FAISS index size: {index.ntotal}")

with open(id_to_filename_path, "r") as f:
    id_to_filename = json.load(f)
logger.debug(f"Number of entries in id_to_filename: {len(id_to_filename)}")
logger.debug(
    f"Sample of id_to_filename mapping: {list(id_to_filename.items())[:5]}")

max_index = max(int(key.split('_')[1]) for key in id_to_filename.keys())
logger.debug(f"Highest index in id_to_filename: {max_index}")


def create_feature_model(num_classes=13):
    resnet50 = models.resnet50(pretrained=False)
    num_features = resnet50.fc.in_features
    resnet50.fc = nn.Sequential(
        nn.Linear(num_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(1024, num_classes)
    )
    return resnet50


def load_model(model_path, num_classes=13, new_num_classes=2048):
    model = create_feature_model(num_classes)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    num_features = model.fc[0].in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(1024, new_num_classes)
    )
    nn.init.xavier_uniform_(model.fc[3].weight)
    nn.init.zeros_(model.fc[3].bias)
    return model


feature_model = load_model(model_path, num_classes=13, new_num_classes=index.d)
feature_model.eval()

logger.debug(
    f"Model loaded. Output dimension: {feature_model.fc[3].out_features}")


def extract_features(image: Image.Image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    image = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = feature_model(image)
    return features.cpu().numpy().flatten()


def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


@app.get('/healthcheck')
def healthcheck():
    return {"message": "API is up and running!"}


@app.post('/predict/feature_embedding')
async def predict_image(image: UploadFile = File(...)):
    contents = await image.read()
    pil_image = Image.open(io.BytesIO(contents)).convert('RGB')
    features = extract_features(pil_image)
    return JSONResponse(content={"features": features.tolist()})


@app.post('/predict/similar_images')
async def predict_combined(image: UploadFile = File(...), top_k: int = Form(5)):
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert('RGB')
        query_features = extract_features(pil_image)

        logger.debug(f"Query features shape: {query_features.shape}")
        logger.debug(f"Sample of query features: {query_features[:5]}")

        query_features = normalize_embeddings(query_features.reshape(1, -1))

        logger.debug(
            f"Query features shape after normalization: {query_features.shape}")
        logger.debug(f"FAISS index size: {index.ntotal}")
        logger.debug(f"FAISS index dimension: {index.d}")
        logger.debug(f"Query features dimension: {query_features.shape[1]}")

        assert query_features.shape[
            1] == index.d, f"Query features dimension ({query_features.shape[1]}) does not match FAISS index dimension ({index.d})"

        distances, indices = index.search(query_features, top_k)

        logger.debug(f"Indices found: {indices}")
        logger.debug(f"Distances: {distances}")

        similar_images = []
        similar_labels = []
        valid_distances = []
        missed_indices = []

        for idx, distance in zip(indices[0], distances[0]):
            str_idx = f"embedding_{idx}"
            if str_idx in id_to_filename:
                similar_images.append(id_to_filename[str_idx])
                similar_labels.append(decoder.get(int(idx), "Unknown"))
                valid_distances.append(float(distance))
            else:
                logger.warning(
                    f"Index {idx} not found in id_to_filename mapping")
                missed_indices.append(int(idx))
                similar_images.append(f"missing_image_{idx}.jpg")
                similar_labels.append("Unknown")
                valid_distances.append(float(distance))

        response_content = {
            "similar_images": similar_images,
            "similar_labels": similar_labels,
            "distances": valid_distances,
            "missed_indices": missed_indices
        }

        logger.info(f"Number of similar images found: {len(similar_images)}")
        logger.debug(f"Response content: {response_content}")
        logger.debug(
            f"Keys in id_to_filename: {list(id_to_filename.keys())[:10]}")
        logger.debug(f"Missed indices: {missed_indices}")

        return JSONResponse(content=response_content)
    except Exception as e:
        logger.exception("An error occurred in predict_combined")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
