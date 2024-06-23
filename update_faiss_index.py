import json
import faiss
import pickle
import numpy as np


def load_embeddings_from_json(json_path):
    embeddings_dict = {}
    chunk_size = 1000  # Adjust based on your chunking strategy
    current_chunk = ""

    with open(json_path, 'r') as f:
        for line in f:
            stripped_line = line.strip()
            if stripped_line.endswith(']'):
                current_chunk += stripped_line
                try:
                    # Attempt to load the current chunk as JSON
                    chunk_dict = json.loads(f"{{{current_chunk}}}")
                    embeddings_dict.update(chunk_dict)
                    current_chunk = ""  # Reset for the next chunk
                except json.JSONDecodeError:
                    # If decoding fails, continue adding lines to the current chunk
                    continue
            else:
                current_chunk += stripped_line

    return embeddings_dict


# Paths to necessary files
index_path = r"C:\Users\nacho\New folder\AiCore\Facebook_Project\index.faiss"
id_to_filename_path = r"C:\Users\nacho\New folder\AiCore\Facebook_Project\id_to_filename.json"
decoder_path = r"C:\Users\nacho\New folder\AiCore\Facebook_Project\image_decoder.pkl"
# Path to your embeddings if needed
embeddings_path = r"C:\Users\nacho\New folder\AiCore\Facebook_Project\image_embeddings.json"

# Load the FAISS index
index = faiss.read_index(index_path)
index_size = index.ntotal

# Load the id_to_filename mapping
with open(id_to_filename_path, 'r') as f:
    id_to_filename = json.load(f)
id_to_filename_size = len(id_to_filename)

# Load the decoder
with open(decoder_path, 'rb') as f:
    decoder = pickle.load(f)
decoder_size = len(decoder)

print(f"FAISS index size: {index_size}")
print(f"Number of entries in id_to_filename: {id_to_filename_size}")
print(f"Number of entries in decoder: {decoder_size}")

# Load embeddings using the chunk processing function
embeddings_dict = load_embeddings_from_json(embeddings_path)
print(f"Number of embeddings: {len(embeddings_dict)}")

# Assuming embeddings_dict is a dictionary with string keys and list values
embeddings = np.array([embeddings_dict[key]
                      for key in sorted(embeddings_dict.keys())])
print(f"Shape of embeddings array: {embeddings.shape}")

correct_size = max(index_size, id_to_filename_size, len(embeddings))
print(f"Target size for all components: {correct_size}")

# Update id_to_filename mapping if necessary
if id_to_filename_size < correct_size:
    print(
        f"Updating id_to_filename from {id_to_filename_size} to {correct_size}")
    for i in range(id_to_filename_size, correct_size):
        id_to_filename[f'image_{i}'] = r'C:\Users\nacho\New folder\AiCore\Facebook_Project\Cleaned_images\00b75a5a-8270-440b-90da-cd2841368acf.jpg'

    with open(id_to_filename_path, 'w') as f:
        json.dump(id_to_filename, f)

# Rebuild FAISS index if necessary
if index_size < correct_size:
    print(f"Rebuilding FAISS index from {index_size} to {correct_size}")
    # Use only the correct number of embeddings
    embeddings = embeddings[:correct_size]

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, index_path)

# Update decoder
if decoder_size < correct_size:
    print(f"Updating decoder from {decoder_size} to {correct_size}")
    for i in range(decoder_size, correct_size):
        decoder[f'image_{i}'] = 'Unknown'  # Use a default label if necessary

    with open(decoder_path, 'wb') as f:
        pickle.dump(decoder, f)

print("FAISS index, id_to_filename mapping, and decoder have been updated.")
