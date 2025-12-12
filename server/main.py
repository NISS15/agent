import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel
from typing import Optional
from pipecatcloud.agent import DailySessionArguments
# Import your bot entrypoint
from bot import bot as bot_entrypoint

load_dotenv(override=True)

app = FastAPI(title="Agentic - Render Ready")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get("FRONTEND_ORIGIN", "*")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SessionRequest(BaseModel):
    # Optional: allow passing a pre-created room_url and token
    room_url: Optional[str] = None
    token: Optional[str] = None

@app.post("/session")
async def create_session(req: SessionRequest):
    """Returns a Daily room url + token (or triggers bot worker)."""
    # This endpoint in the example returns the same structure expected by the frontend.
    # In production, you can create a Daily room here and spawn a worker to run the bot.
    # For the pipecat example, your deployment should have a background worker that runs the bot.
    return {"room_url": req.room_url or None, "token": req.token or None}

@app.get("/")
async def root():
    return {"status": "Agentic Render Ready"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server.main:app", host="0.0.0.0", port=port, reload=False)
