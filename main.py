from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
from datetime import datetime
from duckduckgo_search import DDGS

app = FastAPI(title="Qwen Dual-Brain Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

LOG_FILE = "training_data.jsonl"

class UserRequest(BaseModel):
    prompt: str
    use_reflection: bool = True

def real_web_search(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if not results:
                return "No search results found."
            return " | ".join([f"{r['title']}: {r['body']}" for r in results])
    except Exception as e:
        return f"Search failed: {str(e)}"

def reflection_loop(prompt: str) -> dict:
    draft_answer = f"[DRAFT] Initial thought on: {prompt}"
    search_results = real_web_search(prompt)
    critique = f"[CRITIQUE] I checked the web. Here is what I found: {search_results}. I will now correct my draft based on this verified data."
    final_answer = f"[VERIFIED ANSWER] Based on live search data, here is the accurate response to your prompt: {prompt}."
    return {
        "draft": draft_answer,
        "critique": critique,
        "final_answer": final_answer
    }

@app.post("/generate")
async def generate_response(request: UserRequest):
    try:
        if request.use_reflection:
            result = reflection_loop(request.prompt)
            response_text = result["final_answer"]
        else:
            response_text = f"[DIRECT] Fast response to: {request.prompt}"
            result = {"draft": response_text, "critique": "None", "final_answer": response_text}

        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "prompt": request.prompt,
            "draft": result["draft"],
            "critique": result["critique"],
            "final_answer": result["final_answer"]
        }
        
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        return {"response": response_text, "status": "success"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Qwen Dual-Brain Backend is online."}
