from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
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

fragments = {}
index = None 

def load_faiss_index():
    global index
    if os.path.exists("faiss_index.index"):
        index = faiss.read_index("faiss_index.index")
        print("Índice FAISS cargado correctamente.")
    else:
        print("El archivo 'faiss_index.index' no existe.")


def load_fragments():
    for file in os.listdir("scripts"):
        if file.endswith(".txt"):
            with open(os.path.join("scripts", file), "r", encoding="utf-8") as f:
                content = f.read()
                fragments[file] = content

load_fragments()
load_faiss_index()

async def generate_query_embedding(query: str):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "http://tormenta.ing.puc.cl/api/embed",
                json={"model": "nomic-embed-text", "input": query},
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            embeddings = response.json().get("embeddings")
            
            if embeddings is None or len(embeddings) == 0:
                print("Respuesta de la API:", response.json())
                raise HTTPException(status_code=500, detail="No se pudo generar el embedding.")
            
            embedding = embeddings[0]
            print("Dimensión del embedding generado:", len(embedding))
            return embedding
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail=f"Error de conexión: {exc}")
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=500, detail=f"Error en la respuesta: {exc}")

def get_real_fragment(index):
    file_name = f"{index}.txt"
    return fragments.get(file_name, "Fragmento no encontrado")

def search_documents(query_embedding, k=5):
    if index is None:
        raise HTTPException(status_code=500, detail="El índice FAISS no está cargado.")
    
    query_embedding_array = np.array(query_embedding).reshape(1, -1)
    if query_embedding_array.shape[1] != index.d:
        raise HTTPException(status_code=400, detail="La dimensión del embedding no coincide con la del índice FAISS.")
    
    distances, indices = index.search(query_embedding_array, k)
    return [get_real_fragment(i) for i in indices[0]]

@app.get("/")
async def health_check():
    return "The health check is successful"

@app.post("/chat")
async def chat(request: Request):
    print("Received request on /chat endpoint")  
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
            async with client.stream(
                "POST", 
                "http://tormenta.ing.puc.cl/api/chat", 
                json={"model": model, "messages": messages_with_context}) as response:
                async for line in response.aiter_lines():
                    if line:
                        yield f"data: {line}\n\n"

    return StreamingResponse(generate_response_stream(), media_type="text/event-stream")