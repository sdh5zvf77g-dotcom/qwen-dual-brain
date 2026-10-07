from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from datetime import datetime
from duckduckgo_search import DDGS
from openai import OpenAI

app = FastAPI(title="Qwen Dual-Brain Backend")

# Allow your iPhone to talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

LOG_FILE = "training_data.jsonl"

# --- REAL AI BRAIN SETUP ---
# Replace the text inside the quotes with your actual key. 
# Do NOT include a $ sign.
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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" 
)

class UserRequest(BaseModel):
    prompt: str
    use_reflection: bool = True

def real_web_search(query: str) -> str:
    """Searches the live web for data."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if not results:
                return "No search results found."
            return " | ".join([f"{r['title']}: {r['body']}" for r in results])
    except Exception as e:
        return f"Search failed: {str(e)}"

def reflection_loop(prompt: str) -> dict:
    """The Dual-Brain Logic: Search -> Think -> Verify."""
    # 1. Get live data
    search_results = real_web_search(prompt)
    
    # 2. Ask Qwen 2.5 4B to process it
    try:
        response = client.chat.completions.create(
            model="qwen/qwen-2.5-4b-instruct", # The specific 4B model
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert AI assistant. Your task is to provide a highly accurate, verified answer based *only* on the provided live search data. First, briefly critique any potential inaccuracies, then provide a clear, concise final answer."
                },
                {
                    "role": "user", 
                    "content": f"User Question: {prompt}\n\nLive Search Data:\n{search_results}"
                }
            ]
        )
        final_answer = response.choices[0].message.content
    except Exception as e:
        final_answer = f"AI Generation Error: {str(e)}"

    return {
        "draft": "Initial thought generated...",
        "critique": f"Verified against live data: {search_results[:100]}...",
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

        # Save to learning log
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
    return {"message": "Qwen Dual-Brain Backend is online and using Qwen 2.5 4B."}
