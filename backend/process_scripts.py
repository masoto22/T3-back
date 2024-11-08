import os
from handle_embeddings import process_single_document, store_embeddings_in_faiss

def process_scripts():
    embeddings = []
    for file_name in os.listdir("scripts"):
        if file_name.endswith(".txt"):
            file_path = os.path.join("scripts", file_name)
            print(f"Procesando el guion: {file_name}")
            embeddings.extend(process_single_document(file_path))

    if embeddings:
        store_embeddings_in_faiss(embeddings)
        print(f"Embeddings almacenados en FAISS para {len(embeddings)} fragmentos.")

if __name__ == "__main__":
    process_scripts()