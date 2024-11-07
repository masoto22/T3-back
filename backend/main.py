from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import requests
import faiss
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import os 

app = FastAPI()

@app.get("/")
async def health_check():
    return "The health check is successful"
