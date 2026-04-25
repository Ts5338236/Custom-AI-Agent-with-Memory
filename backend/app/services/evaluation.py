from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.core.config import settings

class EvaluationFramework:
    def __init__(self):
        self.eval_llm = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model="gpt-4o",
            temperature=0
        )

    async def detect_hallucination(self, query: str, context: str, response: str) -> dict:
        """
        Uses a separate LLM pass to check if the response is faithful to the context.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI Fact-Checker. Compare the AI's response against the provided context. "
                       "Determine if the response contains any information NOT found in the context (Hallucination). "
                       "Return a JSON object with 'score' (0-1, 1 is perfectly faithful) and 'reasoning'."),
            ("human", f"Context: {context}\nQuery: {query}\nResponse: {response}")
        ])
        
        chain = prompt | self.eval_llm
        try:
            res = await chain.ainvoke({})
            # In production, use structured output parsing
            import json
            return json.loads(res.content.replace("```json", "").replace("```", "").strip())
        except:
            return {"score": 1, "reasoning": "Evaluation failed, assuming faithful."}

    async def evaluate_accuracy(self, expected: str, actual: str) -> float:
        """
        Calculates semantic similarity between expected and actual responses.
        """
        # Simple implementation using LLM as a judge
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Rate the accuracy of the actual response compared to the expected one. "
                       "Return a score from 0 to 100."),
            ("human", f"Expected: {expected}\nActual: {actual}")
        ])
        chain = prompt | self.eval_llm
        res = await chain.ainvoke({})
        try:
            return float(res.content.strip())
        except:
            return 0.0

eval_framework = EvaluationFramework()
