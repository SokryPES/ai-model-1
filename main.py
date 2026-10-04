import utils.clients
import utils.http_session
import utils.transport
import utils.session
import os
from fastapi import FastAPI
from langchain_ollama import OllamaLLM
from pydantic import BaseModel

app = FastAPI()

# យក Ollama URL ពី Environment Variable (Default: localhost)
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

llm = OllamaLLM(
    base_url=OLLAMA_URL,
    model="qwen2.5:0.5b"
)

class QueryRequest(BaseModel):
    prompt: str

@app.post("/api/v1/agent/chat")
async def chat_with_agent(request: QueryRequest):
    # ហៅ Qwen2.5:0.5b ឱ្យដំណើរការ Task[cite: 1, 3]
    response = llm.invoke(request.prompt)
    return {"status": "success", "result": response}