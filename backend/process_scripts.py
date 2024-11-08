import os
from handle_embeddings import process_single_document, store_embeddings_in_faiss

def process_scripts():
    # Asegúrate de que la carpeta "scripts" existe
    scripts_folder = "scripts"
    if not os.path.exists(scripts_folder):
        print("La carpeta 'scripts' no existe.")
        return

    # Iterar sobre cada archivo en la carpeta "scripts"
    for file_name in os.listdir(scripts_folder):
        if file_name.endswith(".txt"):
            file_path = os.path.join(scripts_folder, file_name)
            print(f"Procesando el guion: {file_name}")

            # Procesar el guion y generar embeddings
            embeddings = process_single_document(file_path)

            # Almacenar los embeddings en FAISS
            index = store_embeddings_in_faiss(embeddings)
            print(f"Embeddings para {file_name} almacenados en FAISS.")

if __name__ == "__main__":
    process_scripts()