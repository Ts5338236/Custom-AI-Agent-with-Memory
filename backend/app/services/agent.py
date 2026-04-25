from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from app.core.config import settings
from app.services.memory import memory_manager
from langchain_core.messages import HumanMessage, AIMessage
from app.services.prompt_builder import prompt_builder
from app.services.tools import tool_registry

class AgentService:
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model="gpt-4o",
            temperature=0
        )
        self.tools = tool_registry.get_all_tools()

    def _get_prompt(self, context: str):
        """Constructs a dynamic prompt with injected context."""
        return ChatPromptTemplate.from_messages([
            ("system", prompt_builder.get_system_prompt(context)),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

    async def execute(self, input_text: str, session_id: str) -> str:
        """
        Executes the agent with hybrid memory injection.
        """
        # 1. Automated Context Injection (Long-term)
        context = await prompt_builder.build_context(input_text, session_id)
        
        # 2. Short-term memory management
        chat_history = memory_manager.get_history(session_id)
        chat_history = prompt_builder.prune_history(chat_history)
        
        # 3. Dynamic Prompt Construction
        prompt = self._get_prompt(context)
        agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)
        
        try:
            # 4. Execute agent
            response = await executor.ainvoke({
                "input": input_text,
                "chat_history": chat_history
            })
            
            output = response["output"]
            
            # 5. Save to short-term memory
            memory_manager.add_message(session_id, HumanMessage(content=input_text))
            memory_manager.add_message(session_id, AIMessage(content=output))
            
            return output
        except Exception as e:
            print(f"Agent Execution Error: {e}")
            return "I encountered an error while processing your request."

agent_service = AgentService()
