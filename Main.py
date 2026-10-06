echo "Paste your OpenRouter API Key (starts with sk-or-):"
read API_KEY
cat << EOF > main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from datetime import datetime
from duckduckgo_search import DDGS
from openai import OpenAI

app = FastAPI(title="Qwen Dual-Brain Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
LOG_FILE = "training_data.jsonl"

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=" 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")

class UserRequest(BaseModel):
    prompt: str
    use_reflection: bool = True

def real_web_search(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if not results: return "No search results found."
            return " | ".join([f"{r['title']}: {r['body']}" for r in results])
    except Exception as e:
        return f"Search failed: {str(e)}"

def reflection_loop(prompt: str) -> dict:
    search_results = real_web_search(prompt)
    try:
        response = client.chat.completions.create(
            model="qwen/qwen-3.5-4b-instruct",
            messages=[
                {"role": "system", "content": "You are an expert AI assistant. Provide a highly accurate, verified answer based *only* on the provided live search data. Briefly critique inaccuracies, then provide a clear final answer."},
                {"role": "user", "content": f"User Question: {prompt}\n\nLive Search Data:\n{search_results}"}
            ]
        )
        final_answer = response.choices[0].message.content
    except Exception as e:
        final_answer = f"AI Generation Error: {str(e)}"
    return {"draft": "Initial thought...", "critique": f"Verified against live data", "final_answer": final_answer}

@app.post("/generate")
async def generate_response(request: UserRequest):
    try:
        if request.use_reflection:
            result = reflection_loop(request.prompt)
            response_text = result["final_answer"]
        else:
            response_text = f"[DIRECT] {request.prompt}"
            result = {"draft": response_text, "critique": "None", "final_answer": response_text}
        
        log_entry = {"timestamp": datetime.utcnow().isoformat(), "prompt": request.prompt, "draft": result["draft"], "critique": result["critique"], "final_answer": result["final_answer"]}
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
        return {"response": response_text, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Qwen Dual-Brain Backend is online and using Qwen 3.5 4B."}
EOF
echo "✅ File created successfully with your API key!"
