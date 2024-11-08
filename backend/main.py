from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
import faiss
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import os
import logging

app = FastAPI()

# https://medium.com/@timnirmal/stream-openai-respond-through-fastapi-to-next-js-f5395f69687c
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

fragments = {}
index = None 

def load_faiss_index():
    global index
    logging.info("Intentando cargar el índice FAISS...")
    if os.path.exists("faiss_index.index"):
        index = faiss.read_index("faiss_index.index")
        logging.info("Índice FAISS cargado correctamente.")
    else:
        logging.warning("El archivo 'faiss_index.index' no existe.")



def load_fragments():
    if not os.path.exists("scripts"):
        logging.info("La carpeta 'scripts' no existe.")
        return
    for file in os.listdir("scripts"):
        if file.endswith(".txt"):
            with open(os.path.join("scripts", file), "r", encoding="utf-8") as f:
                content = f.read()
                fragments[file] = content

load_fragments()
load_faiss_index()

async def generate_query_embedding(query: str):
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                "http://tormenta.ing.puc.cl/api/embed",
                json={"model": "nomic-embed-text", "input": query},
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail=f"Error de conexión: {exc}")
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=500, detail=f"Error en la respuesta: {exc}")

    data = response.json()
    return np.array(data['embeddings'][0], dtype='float32')

def get_real_fragment(index):
    file_name = f"{index}.txt"
    return fragments.get(file_name, "Fragmento no encontrado")

def search_documents(query_embedding, k=5):
    if index is None:
        raise HTTPException(status_code=500, detail="El índice FAISS no está cargado.")
    distances, indices = index.search(np.array([query_embedding]), k)
    return [get_real_fragment(i) for i in indices[0]]

@app.get("/")
async def health_check():
    return "The health check is successful"

@app.post("/chat")
async def chat(request: Request):
    logging.info("Received request on /chat endpoint")  
    data = await request.json()
    model = data.get("model")
    messages = data.get("messages")
    
    if not model or not messages:
        raise HTTPException(status_code=400, detail="Missing request parameters")

    user_message = messages[-1]['content']
    query_embedding = await generate_query_embedding(user_message)
    relevant_fragments = search_documents(query_embedding)
    context = " ".join(relevant_fragments)
    messages_with_context = [{"role": "system", "content": context}] + messages

    async def generate_response_stream():
        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST", 
                    "http://tormenta.ing.puc.cl/api/chat", 
                    json={"model": model, "messages": messages_with_context}
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line:
                            yield f"data: {line}\\n\\n"
            except httpx.RequestError as exc:
                yield f"data: Error de conexión: {exc}\\n\\n"
            except httpx.HTTPStatusError as exc:
                yield f"data: Error en la respuesta: {exc}\\n\\n"

    return StreamingResponse(generate_response_stream(), media_type="text/event-stream")