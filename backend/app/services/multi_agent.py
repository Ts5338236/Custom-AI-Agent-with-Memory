from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.services.tools import tool_registry
from app.services.prompt_builder import prompt_builder
from app.services.memory import memory_manager
from app.core.tracing import tracing_service
import json

class MultiAgentOrchestrator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
    async def run(self, user_input: str, session_id: str):
        trace_id = tracing_service.start_trace(user_input)
        
        # 1. MEMORY AGENT: Fetch relevant context
        tracing_service.log_step(trace_id, "MemoryAgent", "fetch_context")
        context = await prompt_builder.build_context(user_input, session_id)
        chat_history = memory_manager.get_history(session_id)
        
        # 2. PLANNER AGENT: Decompose the request
        tracing_service.log_step(trace_id, "PlannerAgent", "generate_plan")
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
            tracing_service.log_step(trace_id, "PlannerAgent", "plan_created", {"plan": plan})
        except:
            plan = [{"task": user_input}] # Fallback to single task
            tracing_service.log_step(trace_id, "PlannerAgent", "plan_failed", {"fallback": True})

        # 3. EXECUTOR AGENT: Run the tasks
        from app.services.agent import agent_service
        
        final_results = []
        for step in plan:
            task_desc = step.get("task", "")
            tracing_service.log_step(trace_id, "ExecutorAgent", "execute_task", {"task": task_desc})
            
            # Use execute instead of execute_stream for internal steps
            # (Adding a simple execute method to agent_service for this)
            result = await agent_service.execute_internal(task_desc, session_id)
            final_results.append(f"Task: {task_desc}\nResult: {result}")
            tracing_service.log_step(trace_id, "ExecutorAgent", "task_completed", {"result": result[:100] + "..."})

        # 4. FINAL SYNTHESIZER: Combine results
        tracing_service.log_step(trace_id, "SynthesizerAgent", "synthesize_response")
        synth_prompt = ChatPromptTemplate.from_messages([
            ("system", "Synthesize the following task results into a final response for the user."),
            ("human", "Original Query: {query}\n\nTask Results:\n{results}")
        ])
        synth_chain = synth_prompt | self.llm
        synth_response = await synth_chain.ainvoke({
            "query": user_input,
            "results": "\n\n".join(final_results)
        })
        
        # 5. REFLECTION AGENT: Review the response
        tracing_service.log_step(trace_id, "ReviewerAgent", "check_accuracy")
        reviewer_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Quality Reviewer. Compare the AI response to the original query and the task results. "
                       "Is anything missing? Is there any hallucination? "
                       "Return 'PASSED' if it is perfect, or provide a 'CRITIQUE' if it needs fixing."),
            ("human", f"Original Query: {user_input}\nTask Results: {final_results}\nAI Response: {synth_response.content}")
        ])
        
        review = await self.llm.ainvoke(reviewer_prompt.format_messages())
        
        if "PASSED" in review.content:
            tracing_service.log_step(trace_id, "ReviewerAgent", "review_passed")
            final_output = synth_response.content
        else:
            # CORRECTION PASS
            tracing_service.log_step(trace_id, "ReviewerAgent", "review_failed", {"critique": review.content})
            correction_prompt = ChatPromptTemplate.from_messages([
                ("system", f"Your previous response was critiqued: {review.content}\n"
                           "Please provide a corrected version of the final response."),
                ("human", synth_response.content)
            ])
            corrected_response = await self.llm.ainvoke(correction_prompt.format_messages())
            final_output = corrected_response.content
            tracing_service.log_step(trace_id, "SynthesizerAgent", "correction_applied")

        tracing_service.end_trace(trace_id)
        return final_output

multi_agent_orchestrator = MultiAgentOrchestrator()
