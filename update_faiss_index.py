import numpy as np
import faiss
import time
import os
import json


def load_embeddings(file_path):
    print(f"Loading embeddings from {file_path}...")
    data = np.load(file_path)
    embeddings = data['embeddings']
    filenames = data['filenames']
    print(f"Loaded {len(embeddings)} embeddings.")
    return embeddings, filenames


def update_faiss_index(embeddings, filenames, index_path, id_to_filename_path):
    print("Starting FAISS index update process...")
    start_time = time.time()

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms

    # Create and populate FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(normalized_embeddings)

    print(f"FAISS index created with {index.ntotal} vectors.")

    # Save the updated index
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to {index_path}")

    # Update id_to_filename mapping
    id_to_filename = {i: filename for i, filename in enumerate(filenames)}
    with open(id_to_filename_path, 'w') as f:
        json.dump(id_to_filename, f)
    print(f"id_to_filename mapping saved to {id_to_filename_path}")

    total_time = time.time() - start_time
    print(f"FAISS index update completed in {total_time:.2f} seconds.")


# Paths
current_directory = os.path.dirname(os.path.abspath(__file__))
embeddings_path = os.path.join(current_directory, 'embeddings.npz')
index_path = os.path.join(current_directory, 'index.faiss')
id_to_filename_path = os.path.join(current_directory, 'id_to_filename.json')

# Load embeddings
embeddings, filenames = load_embeddings(embeddings_path)

# Update FAISS index
update_faiss_index(embeddings, filenames, index_path, id_to_filename_path)

print("Script completed.")
