from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import requests
import faiss
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import os 

app = FastAPI()

# https://medium.com/@timnirmal/stream-openai-respond-through-fastapi-to-next-js-f5395f69687c
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# Cargar el índice FAISS
index = faiss.read_index("faiss_index.index")

# Cargar los fragmentos desde los archivos
fragments = {}

def load_fragments():
    for file in os.listdir("scripts"):
        if file.endswith(".txt"):
            with open(os.path.join("scripts", file), "r", encoding="utf-8") as f:
                content = f.read()
                fragments[file] = content  # Usa el nombre del archivo como clave

load_fragments()

# Generación de embedding para la consulta
def generate_query_embedding(query: str):
    response = requests.post(
        "http://tormenta.ing.puc.cl/api/embed",
        json={"model": "nomic-embed-text", "input": query},
        headers={'Content-Type': 'application/json'}
    )
    if response.status_code == 200:
        data = response.json()
        return np.array(data['embeddings'][0], dtype='float32')
    else:
        return JSONResponse(status_code=500, content={"detail": "Error al generar embedding para la consulta"})

def get_real_fragment(index):
    # Suponiendo que el índice es el nombre del archivo sin la extensión
    file_name = f"{index}.txt"  # Ajusta esto según tu lógica de indexación
    return fragments.get(file_name, "Fragmento no encontrado")

def search_documents(query_embedding, k=5):
    distances, indices = index.search(np.array([query_embedding]), k)
    # Retornar fragmentos reales
    return [get_real_fragment(i) for i in indices[0]]

@app.post("/complete")
async def complete(request: Request):
    data = await request.json()
    prompt = data.get("prompt")
    if not prompt:
        return JSONResponse(status_code=400, content={"detail": "No se proporcionó un prompt"})
    payload = {
        "model": "integra-LLM",
        "prompt": prompt,
        "temperature": 0.6,
        "max_tokens": 150
    }

    try:
        response = requests.post(
            "http://tormenta.ing.puc.cl/api/complete", 
            json=payload, 
            headers={'Content-Type': 'application/json'}
            )
        response.raise_for_status()
        result = response.json()
        return JSONResponse(content=result)
    except requests.exceptions.RequestException as e:
        return JSONResponse(status_code=500, content={"detail": f"Error al comunicarse con la API del LLM: {e}"})


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    model = data.get("model")
    messages = data.get("messages")
    
    if not model or not messages:
        return JSONResponse(status_code=400, content={"detail": "Faltan parámetros en la solicitud"})

    async def generate_response_stream():
        context = " ".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        response = requests.post(
            "http://tormenta.ing.puc.cl/api/chat",
            json={"model": model, "messages": messages},
            headers={'Content-Type': 'application/json'},
            stream=True
        )
        for line in response.iter_lines():
            if line:
                yield f"data: {line.decode('utf-8')}\n\n"

    return StreamingResponse(generate_response_stream(), media_type="text/event-stream")

@app.post("/search")
async def search(query: str):
    try:
        query_embedding = generate_query_embedding(query)
        fragments = search_documents(query_embedding)
        context = " ".join(fragments)
        
        payload = {
            "model": "integra-LLM",
            "prompt": f"{context}\n\nPregunta: {query}",
            "temperature": 0.6,
            "max_tokens": 150,
            "format": "json"
        }
        print(f"prompt enviado al llm: {payload['prompt']}")
        response = requests.post(
            "http://tormenta.ing.puc.cl/api/complete", 
            json=payload, 
            headers={'Content-Type': 'application/json'}
            )
        
        if response.status_code == 200:
            return JSONResponse(content=response.json())
        else:
            return JSONResponse(status_code=500, content={"detail": "Error al obtener respuesta del LLM"})
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})

