import pickle
import fastapi
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from fastapi import File
from fastapi import UploadFile
from fastapi import Form
import torch
import torch.nn as nn
from pydantic import BaseModel
from torchvision import models
import image_processor


class FeatureExtractor(nn.Module):
    def __init__(self,
                 decoder: dict = None):
        super(FeatureExtractor, self).__init__()
        self.main = models.resnet50(pretrained=True)

        for param in self.main.parameters():
            param.requires_grad = False
        for param in self.main.layer3.parameters():
            param.requires_grad = True
        for param in self.main.layer4.parameters():
            param.requires_grad = True
        self.decoder = decoder

        num_features = self.main.fc.in_features
        num_categories = len(decoder)
        self.main.fc = nn.Linear(num_features, num_categories)
        self.decoder = decoder

    def forward(self, image):
        x = self.main(image)
        return x

    def predict(self, image):
        with torch.no_grad():
            x = self.forward(image)
            return x

# Don't change this, it will be useful for one of the methods in the API


class TextItem(BaseModel):
    text: str


try:
    model_path = r"C:\Users\nacho\New folder\AiCore\Facebook_Project\model_evaluation\weights_20240405_013955\final_weights.pth"
    decoder_path = "path/to/your/image_decoder.pkl"

    with open(decoder_path, "rb") as f:
        decoder = pickle.load(f)

    model = FeatureExtractor(decoder=decoder)
    model.load_state_dict(torch.load(model_path))
    pass
except:
    raise OSError(
        "No Feature Extraction model found. Check that you have the decoder and the model in the correct location")

try:
    faiss_index_path = r"C:\Users\nacho\New folder\AiCore\Facebook_Project\faiss_search.py"
    with open(faiss_index_path, "rb") as f:
        faiss_index = pickle.load(f)
    pass
except:
    raise OSError(
        "No Image model found. Check that you have the encoder and the model in the correct location")


app = FastAPI()
print("Starting server")


@app.get('/healthcheck')
def healthcheck():
    msg = "API is up and running!"

    return {"message": msg}


@app.post('/predict/feature_embedding')
def predict_image(image: UploadFile = File(...)):
    pil_image = Image.open(image.file)
    processed_image = image_processor.preprocess(pil_image)
    features = model.predict(processed_image)
    return JSONResponse(content={"features": features.tolist()})


@app.post('/predict/similar_images')
def predict_combined(image: UploadFile = File(...), text: str = Form(...)):
    print(text)
    pil_image = Image.open(image.file)
    processed_image = image_processor.preprocess(pil_image)
    features = model.predict(processed_image)
    similar_indexes, _ = faiss_index.search(features, k=10)
    similar_images = [model.decoder[index] for index in similar_indexes[0]]
    return JSONResponse(content={"similar_index": similar_images})


if __name__ == '__main__':
    uvicorn.run("api:app", host="0.0.0.0", port=8080)
