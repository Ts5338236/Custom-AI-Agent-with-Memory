import pytest
from app.services.evaluation import eval_framework
from app.services.agent import agent_service

@pytest.mark.asyncio
async def test_agent_accuracy():
    """
    Test case: Verify the agent correctly recalls a specific fact.
    """
    test_query = "What is the capital of France?"
    expected_answer = "The capital of France is Paris."
    
    # Run the agent
    actual_response = await agent_service.execute_stream(test_query, "test_session")
    
    # For testing, we collect the full response from the generator
    full_response = ""
    async for token in actual_response:
        full_response += token
        
    # Evaluate
    score = await eval_framework.evaluate_accuracy(expected_answer, full_response)
    
    assert score > 80, f"Accuracy score {score} is too low for a simple fact check."

@pytest.mark.asyncio
async def test_hallucination_detection():
    """
    Test case: Ensure the agent doesn't hallucinate info not in context.
    """
    context = "The user's favorite book is 'The Hobbit'."
    query = "What is my favorite book?"
    response = "Your favorite book is 'The Lord of the Rings'." # Hallucination!
    
    result = await eval_framework.detect_hallucination(query, context, response)
    
    assert result["score"] < 0.5, "Hallucination was not detected correctly."
