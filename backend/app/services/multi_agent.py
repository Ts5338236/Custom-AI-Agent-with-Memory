from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.services.tools import tool_registry
from app.services.prompt_builder import prompt_builder
from app.services.memory import memory_manager
import json

class MultiAgentOrchestrator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
    async def run(self, user_input: str, session_id: str):
        # 1. MEMORY AGENT: Fetch relevant context
        context = await prompt_builder.build_context(user_input, session_id)
        chat_history = memory_manager.get_history(session_id)
        
        # 2. PLANNER AGENT: Decompose the request
        planner_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Planner Agent. Break the user's request into 1-3 specific steps. "
                       "Context: {context}\n"
                       "Available Tools: {tools}\n"
                       "Return a JSON list of tasks."),
            ("human", "{input}")
        ])
        
        tools_desc = "\n".join([f"- {t.name}: {t.description}" for t in tool_registry.get_all_tools()])
        planner_chain = planner_prompt | self.llm
        plan_response = await planner_chain.ainvoke({
            "context": context,
            "tools": tools_desc,
            "input": user_input
        })
        
        # Parse plan (naive)
        try:
            plan = json.loads(plan_response.content.replace("```json", "").replace("```", "").strip())
        except:
            plan = [{"task": user_input}] # Fallback to single task

        # 3. EXECUTOR AGENT: Run the tasks
        # For simplicity, we use our existing AgentService logic as the "Executor"
        from app.services.agent import agent_service
        
        final_results = []
        for step in plan:
            task_desc = step.get("task", "")
            result = await agent_service.execute(task_desc, session_id)
            final_results.append(f"Task: {task_desc}\nResult: {result}")

        # 4. FINAL SYNTHESIZER: Combine results
        synth_prompt = ChatPromptTemplate.from_messages([
            ("system", "Synthesize the following task results into a final response for the user."),
            ("human", "Original Query: {query}\n\nTask Results:\n{results}")
        ])
        synth_chain = synth_prompt | self.llm
        final_response = await synth_chain.ainvoke({
            "query": user_input,
            "results": "\n\n".join(final_results)
        })
        
        return final_response.content

multi_agent_orchestrator = MultiAgentOrchestrator()
