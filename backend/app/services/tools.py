from langchain.tools import tool
import datetime

@tool
def get_current_time(query: str) -> str:
    """Returns the current server time. Useful for scheduling or checking dates."""
    return f"The current server time is {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."

@tool
def get_weather(location: str) -> str:
    """
    Returns the current weather for a given location.
    Input should be a city name (e.g., 'San Francisco').
    """
    # In a real app, this would call a weather API like OpenWeatherMap
    return f"The weather in {location} is currently 72°F and sunny."

@tool
def google_search(query: str) -> str:
    """
    Searches the web for information. Use this for general knowledge or current events.
    """
    # Mocking a search result
    return f"Search results for '{query}': [1] Result 1 from source A... [2] Result 2 from source B..."

from app.services.vector_db import vector_db_service

@tool
def search_long_term_memory(query: str) -> str:
    """
    Searches the long-term memory for relevant past conversations or facts.
    Use this if the user asks about something from a long time ago or a general preference.
    """
    results = vector_db_service.search_memories(query, k=3)
    if not results:
        return "No relevant memories found."
    
    formatted = "\n".join([f"- {r['content']}" for r in results])
    return f"Relevant past memories:\n{formatted}"

from app.services.memory_intelligence import intel_memory_manager

@tool
async def save_to_long_term_memory(text: str) -> str:
    """
    Saves a new fact or important piece of information to the long-term memory.
    This tool automatically categorizes and scores the information for better future recall.
    """
    result = await intel_memory_manager.process_and_save(text)
    return f"Information processed: {result}"

@tool
def send_email(recipient: str, subject: str, body: str) -> str:
    """
    Sends an email to a specific recipient. 
    Use this for notifications, summaries, or direct communication.
    """
    # Mock integration with SendGrid/AWS SES
    return f"Email successfully sent to {recipient} with subject: {subject}."

@tool
def manage_calendar(action: str, details: str) -> str:
    """
    Manages user calendar events. Action can be 'create', 'list', or 'delete'.
    Details should include date/time and event description.
    """
    # Mock integration with Google Calendar API
    return f"Calendar event '{details}' successfully processed with action: {action}."

@tool
def process_payment(amount: float, currency: str, description: str) -> str:
    """
    Processes a financial transaction. Use this for payments or invoices.
    """
    # Mock integration with Stripe API
    transaction_id = "txn_" + "".join([str(i) for i in range(5)])
    return f"Payment of {amount} {currency} for '{description}' processed. ID: {transaction_id}"

@tool
def extract_text_from_image(image_url: str) -> str:
    """
    Performs OCR (Optical Character Recognition) on an image URL.
    Use this when the user provides an image or document for analysis.
    """
    # Mock integration with AWS Textract or Google Vision API
    return "Extracted text: [INVOICE] Date: 2026-04-25. Total Amount: $150.00. Vendor: Acme Corp."

class ToolRegistry:
    def __init__(self):
        # Register all tools here
        self._tools = [
            get_current_time,
            get_weather,
            google_search,
            search_long_term_memory,
            save_to_long_term_memory,
            send_email,
            manage_calendar,
            process_payment,
            extract_text_from_image
        ]

    def get_all_tools(self):
        return self._tools

    def add_tool(self, tool_func):
        """Allows dynamic addition of tools."""
        self._tools.append(tool_func)

tool_registry = ToolRegistry()
