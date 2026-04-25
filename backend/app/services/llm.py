from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.core.config import settings
from app.schemas.chat import ChatMessage
from typing import List

class LLMService:
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model="gpt-4o",
            temperature=0.7
        )

    async def get_chat_response(self, message: str, history: List[ChatMessage] = []) -> str:
        messages = [SystemMessage(content="You are a helpful AI Assistant with memory.")]
        
        # Add history to context
        for msg in history:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            else:
                messages.append(AIMessage(content=msg.content))
        
        # Add current message
        messages.append(HumanMessage(content=message))
        
        try:
            response = await self.llm.ainvoke(messages)
            return response.content
        except Exception as e:
            # In a production app, log this properly
            print(f"Error calling LLM: {e}")
            raise Exception("Failed to get response from AI service.")

llm_service = LLMService()
