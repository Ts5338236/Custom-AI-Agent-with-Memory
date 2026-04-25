from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy.orm import Session
from app.models.chat import User
from app.core.config import settings
import json

class PersonalizationService:
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model="gpt-4o",
            temperature=0
        )

    async def extract_preferences(self, user_input: str, current_prefs: dict) -> dict:
        """
        Analyzes user input to see if any new preferences or facts can be learned.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a User Profile Expert. Analyze the user's message. "
                       "If you find any personal preferences, interests, or facts, "
                       "update the current preferences JSON and return it. "
                       "If nothing new is found, return the current preferences. "
                       "Current Preferences: {current_prefs}"),
            ("human", "{input}")
        ])
        
        chain = prompt | self.llm
        try:
            response = await chain.ainvoke({"input": user_input, "current_prefs": json.dumps(current_prefs)})
            # Extract JSON from response
            new_prefs = json.loads(response.content.replace("```json", "").replace("```", "").strip())
            return new_prefs
        except Exception as e:
            print(f"Preference Extraction Error: {e}")
            return current_prefs

    async def update_user_profile(self, db: Session, user_id: int, user_input: str):
        """
        Orchestrates the profile update process.
        """
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return

        new_prefs = await self.extract_preferences(user_input, user.preferences or {})
        
        if new_prefs != user.preferences:
            user.preferences = new_prefs
            db.commit()

personalization_service = PersonalizationService()
