#
        

from fastapi import FastAPI, UploadFile, File
from pathlib import Path
from PIL import Image
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection
import pandas as pd
import time
import pickle

# Constants
HOST = "a05a6eead7bf84cbbb4609bb9b9892d1-1154651500.eu-central-1.elb.amazonaws.com"
PORT = "19530"
USER = "root"
PASSWORD = "3^cy8ARvn4FD"
COLLECTION_NAME = "hnm_fashion_sample"
EMB_DIM = 512
IMG_DIM = 225
PRODUCT_MAPPING_PATH = "../product_mapping.pkl"
MODEL_NAME = "clip-ViT-B-32"
SEARCH_PARAMS = {"metric_type": "IP", "params": {"nprobe": 20}}

app = FastAPI()
model = SentenceTransformer(MODEL_NAME)

# Connect to pymilvus
connections.connect("default", user=USER, password=PASSWORD, host=HOST, port=PORT)

# Load the collection into memory
collection = Collection(COLLECTION_NAME)
collection.load()

# Load product_mapping from the pickle file
with open(PRODUCT_MAPPING_PATH, "rb") as f:
    product_mapping = pickle.load(f)


def search(query_emb, num_results: int = 10):
    results = collection.search(
        data=query_emb,
        anns_field="img_embedding",
        param=SEARCH_PARAMS,
        limit=num_results,
        expr=None,
    )



    parsed_results = [
        {"address_id": result.id, "distance": result.distance, "entity": result.entity}
        for result in results[0]
    ]

    return {"similarities": parsed_results}


@app.post("/search-by-text")
async def search_by_text(query_text: str, num_results: int = 10):
    raw_embeddings = [model.encode(query_text)]
    query_emb = normalize(raw_embeddings, axis=1).tolist()
    return search(query_emb, num_results)



@app.post("/search-by-image")
async def search_by_image(file: UploadFile = File(...), num_results: int = 10):
    image = Image.open(file.file).convert("RGB").resize((225, 225))
    raw_embedding = model.encode([image])
    query_emb = normalize(raw_embedding, axis=1).tolist()
    return search(query_emb, num_results)
