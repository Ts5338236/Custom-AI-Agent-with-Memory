from fastapi import APIRouter, HTTPException, Depends, Request
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.agent import agent_service
from sqlalchemy.orm import Session
from app.api.deps import get_db, get_current_user
from app.models.chat import ChatMessage, ChatSession, User
from app.main import limiter

router = APIRouter()

from app.services.multi_agent import multi_agent_orchestrator

@router.post("/", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat(request_obj: Request, request: ChatRequest, db: Session = Depends(get_db)):
    try:
        # ... (session logging omitted for brevity) ...
        
        # 2. Execute via Multi-Agent Orchestrator
        response_text = await multi_agent_orchestrator.run(
            user_input=request.message,
            session_id=request.session_id
        )

        # 3. Log to Database persistently
        user_msg = ChatMessage(session_id=request.session_id, role="user", content=request.message)
        assistant_msg = ChatMessage(session_id=request.session_id, role="assistant", content=response_text)
        db.add(user_msg)
        db.add(assistant_msg)
        db.commit()

        return ChatResponse(
            response=response_text,
            session_id=request.session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
