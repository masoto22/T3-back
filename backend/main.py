from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
import faiss
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

index = faiss.read_index("faiss_index.index")
fragments = {}

def load_fragments():
    for file in os.listdir("scripts"):
        if file.endswith(".txt"):
            with open(os.path.join("scripts", file), "r", encoding="utf-8") as f:
                content = f.read()
                fragments[file] = content

load_fragments()

async def generate_query_embedding(query: str):
    async with httpx.AsyncClient() as client:
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

@app.post("/complete")
async def complete(request: Request):
    data = await request.json()
    prompt = data.get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt not provided")

    payload = {
        "model": "integra-LLM",
        "prompt": prompt,
        "temperature": 0.6,
        "max_tokens": 150
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "http://tormenta.ing.puc.cl/api/complete",
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            return JSONResponse(content=response.json())
        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Error contacting LLM API: {e}")

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    model = data.get("model")
    messages = data.get("messages")
    
    if not model or not messages:
        raise HTTPException(status_code=400, detail="Missing request parameters")

    async def generate_response_stream():
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://tormenta.ing.puc.cl/api/chat",
                json={"model": model, "messages": messages},
                headers={'Content-Type': 'application/json'},
                stream=True
            )
            async for line in response.aiter_lines():
                if line:
                    yield f"data: {line}\n\n"

    return StreamingResponse(generate_response_stream(), media_type="text/event-stream")

@app.post("/search")
async def search(query: str):
    try:
        query_embedding = await generate_query_embedding(query)
        fragments = search_documents(query_embedding)
        context = " ".join(fragments)
        
        payload = {
            "model": "integra-LLM",
            "prompt": f"{context}\n\nPregunta: {query}",
            "temperature": 0.6,
            "max_tokens": 150,
            "format": "json"
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://tormenta.ing.puc.cl/api/complete", 
                json=payload, 
                headers={'Content-Type': 'application/json'}
            )
        
        if response.status_code == 200:
            return JSONResponse(content=response.json())
        else:
            raise HTTPException(status_code=500, detail="Error fetching response from LLM")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


