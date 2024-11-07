import os
import requests
import logging
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)

# https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/
def load_and_split_documents():
    documents = []
    for file in os.listdir("scripts"):
        with open(f"scripts/{file}", "r", encoding="utf-8") as f:
            content = f.read()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            fragments = splitter.split_text(content)
            documents.extend(fragments)
    return documents

def generate_embeddings(documents):
    embeddings = []
    for doc in documents:
        response = requests.post(
            "http://tormenta.ing.puc.cl/api/embed", 
            json={"model": "nomic-embed-text", "input": doc}, 
            headers={'Content-Type': 'application/json'}
            )
        if response.status_code == 200:
            data = response.json()
            embeddings.append(data['embeddings'][0])
        else:
            logging.error(f"Error al generar embedding: {response.status_code} para el documento: {doc[:30]}...")
            embeddings.append(None)
    return embeddings

# https://medium.com/loopio-tech/how-to-use-faiss-to-build-your-first-similarity-search-bf0f708aa772
def store_embeddings_in_faiss(embeddings):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype='float32'))
    faiss.write_index(index, "faiss_index.index")
    logging.info(f"Almacenados {len(embeddings)} embeddings en FAISS.")
    return index

def main():
    documents = load_and_split_documents()
    embeddings = generate_embeddings(documents)
    store_embeddings_in_faiss(embeddings)

if __name__ == "__main__":
    main()

