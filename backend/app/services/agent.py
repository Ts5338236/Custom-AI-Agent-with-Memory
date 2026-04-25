from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from app.core.config import settings
from app.services.memory import memory_manager
from langchain_core.messages import HumanMessage, AIMessage
from app.services.prompt_builder import prompt_builder
from app.services.tools import tool_registry
from app.core.resilience import standard_retry, llm_breaker

class AgentService:
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model="gpt-4o",
            temperature=0
        )
        self.tools = tool_registry.get_all_tools()

    def _get_prompt(self, context: str, preferences: dict = {}):
        """Constructs a dynamic prompt with injected context and preferences."""
        return ChatPromptTemplate.from_messages([
            ("system", prompt_builder.get_system_prompt(context, preferences)),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

    @llm_breaker
    @standard_retry
    async def execute_stream(self, input_text: str, session_id: str, preferences: dict = {}):
        """
        Executes the agent and yields tokens in real-time.
        """
        context = await prompt_builder.build_context(input_text, session_id)
        chat_history = memory_manager.get_history(session_id)
        chat_history = prompt_builder.prune_history(chat_history)
        
        prompt = self._get_prompt(context, preferences)
        
        try:
            full_response = ""
            async for chunk in self.llm.astream(prompt.format_messages(
                input=input_text, 
                chat_history=chat_history,
                agent_scratchpad=[] 
            )):
                content = chunk.content
                full_response += content
                yield content
            
            # Save to memory after stream finishes
            memory_manager.add_message(session_id, HumanMessage(content=input_text))
            memory_manager.add_message(session_id, AIMessage(content=full_response))
            
        except Exception as e:
            yield f"Error: {str(e)}"

    async def execute_internal(self, input_text: str, session_id: str) -> str:
        """
        Executes the agent and returns the full result as a string.
        Used internally by the orchestrator.
        """
        context = await prompt_builder.build_context(input_text, session_id)
        chat_history = memory_manager.get_history(session_id)
        chat_history = prompt_builder.prune_history(chat_history)
        
        prompt = self._get_prompt(context)
        agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)
        
        try:
            response = await executor.ainvoke({
                "input": input_text,
                "chat_history": chat_history
            })
            return response["output"]
        except Exception as e:
            return f"Error: {str(e)}"

agent_service = AgentService()
