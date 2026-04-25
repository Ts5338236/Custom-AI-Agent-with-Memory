from fastapi import APIRouter, HTTPException, Depends, Request
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.agent import agent_service
from sqlalchemy.orm import Session
from app.api.deps import get_db, get_current_user
from app.models.chat import ChatMessage, ChatSession, User
from app.main import limiter

router = APIRouter()

from app.services.multi_agent import multi_agent_orchestrator

from fastapi.responses import StreamingResponse
from app.services.cache import response_cache
import json

@router.post("/stream")
@limiter.limit("10/minute")
async def chat_stream(request_obj: Request, request: ChatRequest, db: Session = Depends(get_db)):
    # 1. Check Cache
    cached_response = response_cache.get(request.message, request.session_id)
    if cached_response:
        async def generate_cached():
            yield cached_response
        return StreamingResponse(generate_cached(), media_type="text/event-stream")

    # 2. Execute Stream
    async def event_generator():
        full_response = ""
        async for token in agent_service.execute_stream(request.message, request.session_id):
            full_response += token
            yield f"data: {json.dumps({'token': token})}\n\n"
        
        # Cache the full response
        response_cache.set(request.message, request.session_id, full_response)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
