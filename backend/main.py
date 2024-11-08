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
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

fragments = {}
index = None  # Inicializa el índice

def load_fragments():
    for file in os.listdir("scripts"):
        if file.endswith(".txt"):
            with open(os.path.join("scripts", file), "r", encoding="utf-8") as f:
                content = f.read()
                fragments[file] = content

def load_faiss_index():
    global index
    index = faiss.read_index("faiss_index.index")  # Carga el índice de FAISS

load_fragments()
load_faiss_index()

async def generate_query_embedding(query: str):
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://tormenta.ing.puc.cl/api/embed",
            json={"model": "nomic-embed-text", "input": query},
            headers={'Content-Type': 'application/json'}
        )
    if response.status_code == 200:
        data = response.json()
        return np.array(data['embeddings'][0], dtype='float32')
    else:
        raise HTTPException(status_code=500, detail="Error generating embedding")

def get_real_fragment(index):
    file_name = f"{index}.txt"
    return fragments.get(file_name, "Fragmento no encontrado")

def search_documents(query_embedding, k=5):
    distances, indices = index.search(np.array([query_embedding]), k)
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

    # Obtener el último mensaje del usuario
    user_message = messages[-1]['content']
    
    # Generar embedding para el mensaje del usuario
    query_embedding = await generate_query_embedding(user_message)
    
    # Buscar documentos relevantes
    relevant_fragments = search_documents(query_embedding)
    context = " ".join(relevant_fragments)

    # Agregar el contexto a los mensajes
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