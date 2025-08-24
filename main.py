from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import requests

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_API_URL = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"

class SummarizeRequest(BaseModel):
    text: str
    max_length: int = 130
    min_length: int = 30

@app.post("/summarize")
def summarize(request: SummarizeRequest):
    if len(request.text.strip()) < 50:
        raise HTTPException(status_code=400, detail="Text too short to summarize")

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": request.text,
        "parameters": {
            "max_length": request.max_length,
            "min_length": request.min_length,
            "do_sample": False
        }
    }

    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return {"summary": result[0]['summary_text']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/")
def home():
    return {"message": "AI Summarizer API is live! Use POST /summarize"}
