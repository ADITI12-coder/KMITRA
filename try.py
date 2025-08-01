import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the cleaned JSON file
with open("cleaned_data.json", "r", encoding="utf-8") as f:
    cleaned_data = json.load(f)

# Initialize embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Extract text content for embedding
texts = [entry["content"] for entry in cleaned_data if entry["content"].strip()]
urls = [entry["url"] for entry in cleaned_data if entry["content"].strip()]

# Generate embeddings
embeddings = model.encode(texts, convert_to_numpy=True)

# Create a FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save FAISS index and metadata
faiss.write_index(index, "faiss_index.bin")
with open("faiss_metadata.json", "w", encoding="utf-8") as f:
    json.dump({"texts": texts, "urls": urls}, f, indent=4, ensure_ascii=False)

print("FAISS index and metadata saved.")