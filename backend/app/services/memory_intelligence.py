from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.services.vector_db import vector_db_service
from app.core.config import settings
import json

class IntelligenceMemoryManager:
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model="gpt-4o",
            temperature=0
        )
        
    async def process_and_save(self, text: str):
        """
        Analyzes the text, categorizes it, scores importance, and checks for duplicates.
        """
        # 1. Deduplication Check
        existing = vector_db_service.search_memories(text, k=1)
        if existing and existing[0]["score"] > 0.85: # High similarity threshold
            return f"Memory already exists: {existing[0]['content']}"

        # 2. Intelligence Analysis (Categorization & Scoring)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Analyze the following text and return a JSON object with: "
                       "'category' (one of: fact, preference, task, relationship), "
                       "'importance' (1-10 integer based on long-term value), "
                       "'summary' (concise version of the fact)."),
            ("human", "{text}")
        ])
        
        try:
            chain = prompt | self.llm
            response = await chain.ainvoke({"text": text})
            # Naive JSON extraction (in production use a structured output parser)
            analysis = json.loads(response.content.replace("```json", "").replace("```", "").strip())
            
            # 3. Save to Vector DB with metadata
            vector_db_service.add_memory(
                text=analysis.get("summary", text),
                category=analysis.get("category", "fact"),
                importance=analysis.get("importance", 5)
            )
            return f"Saved as {analysis.get('category')} (Importance: {analysis.get('importance')})"
        except Exception as e:
            print(f"Intelligence Memory Analysis Error: {e}")
            # Fallback to simple save
            vector_db_service.add_memory(text)
            return "Saved with fallback (basic fact)."

intel_memory_manager = IntelligenceMemoryManager()
