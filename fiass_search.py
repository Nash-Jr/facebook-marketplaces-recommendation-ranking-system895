import json
import numpy as np
import faiss

with open('image_embeddings.json', 'r') as f:
    image_embeddings = json.load(f)


embeddings = np.array([emb for emb in image_embeddings.values()])


index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)


def search_similar_images(query_image_path, top_k=5):
    query_embedding = extract_image_features(query_image_path)
    distances, indices = index.search(query_embedding.reshape(1, -1), top_k)
    similar_image_ids = [list(image_embeddings.keys())[idx]
                         for idx in indices[0]]
    return similar_image_ids


query_image_path = 'path/to/your/query/image.jpg'
similar_image_ids = search_similar_images(query_image_path, top_k=5)
print(similar_image_ids)
