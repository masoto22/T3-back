import os
import requests
import logging
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)

# https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/
def load_and_split_document(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        fragments = splitter.split_text(content)
    return fragments

def generate_embedding(doc):
    response = requests.post(
        "http://tormenta.ing.puc.cl/api/embed", 
        json={"model": "nomic-embed-text", "input": doc}, 
        headers={'Content-Type': 'application/json'}
    )
    if response.status_code == 200:
        data = response.json()
        return data['embeddings'][0]
    else:
        logging.error(f"Error al generar embedding: {response.status_code} para el documento.")
        return None

def process_single_document(file_path):
    fragments = load_and_split_document(file_path)
    embeddings = []
    print(f"Procesando el documento: {file_path}")
    for i, fragment in enumerate(fragments):
        embedding = generate_embedding(fragment)
        if embedding is not None:
            embeddings.append(embedding)
            print(f"Fragmento {i + 1}/{len(fragments)} procesado y embedding generado.")
    print(f"Total de embeddings generados para {file_path}: {len(embeddings)}")
    return embeddings

# https://medium.com/loopio-tech/how-to-use-faiss-to-build-your-first-similarity-search-bf0f708aa772
def store_embeddings_in_faiss(embeddings):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype='float32'))
    faiss.write_index(index, "faiss_index.index")
    logging.info(f"Almacenados {len(embeddings)} embeddings en FAISS.")
    print(f"Embeddings almacenados en FAISS.")
    return index



