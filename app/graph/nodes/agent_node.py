# app/graph/nodes/agent_node.py

import asyncio
import json
import logging
from app.graph.state import GraphState
from app.services.prompt_service import prompt_service
from app.core.config import settings
from app.services.structured_output_adapter import get_structured_output
from app.services.schemas import AgentOutput
from app.utils.exceptions import AgentExecutionError
from app.graph.nodes.execution_nodes import execute_code_in_sandbox

def create_agent_node(node_id: str):
    """Factory for creating a robust agent node."""
    async def agent_node(state: GraphState) -> dict:
        """Represents a single agent's turn, using the structured output adapter."""
        logging.info(f"--- [FORWARD PASS] Agent: {node_id} | Epoch: {state['epoch']} ---")
                
        try:
            layer_idx, agent_idx = map(int, node_id.split('_')[1:])
            agent_prompt = state['all_layers_prompts'][layer_idx][agent_idx]
        except (ValueError, IndexError) as e:
            logging.error(f"AGENT_NODE_ERROR: Could not find prompt for {node_id}. Error: {e}")
            return {}

        if layer_idx == 0:
            input_data = state["decomposed_problems"].get(node_id, state["original_request"])
        else:
            prev_layer_outputs = [v for k, v in state["agent_outputs"].items() if k.startswith(f"agent_{layer_idx-1}_")]
            input_data = json.dumps(prev_layer_outputs, indent=2)

        agent_memory = state.get("memory", {}).get(node_id, [])
        if len(json.dumps(agent_memory)) > 450000:
             logging.warning(f"Memory for agent {node_id} is large. Summarizing...")
             summarizer_llm = state.get("summarizer_llm")
             summarizer_chain = prompt_service.create_chain(summarizer_llm, "memory_summarizer")
             summary = await summarizer_chain.ainvoke({"history": json.dumps(agent_memory)})
             agent_memory = [{"summary_of_past_epochs": summary}]

        memory_str = json.dumps(agent_memory, indent=2)
        
        prompt_template = """{system_prompt}

### Context
This is a log of your previous proposed solutions and reasonings (memory), and data from the previous layer of agents. Use this to inform your response.

#### Memory
{memory}

#### Input Data
{input_data}
"""
        prompt_input = {
            "system_prompt": agent_prompt,
            "memory": memory_str,
            "input_data": input_data
        }
        
        logging.info(f"--- AGENT {node_id} FULL INPUT ---\n{prompt_template.format(**prompt_input)}")

        max_attempts = 2
        response_object = None
        
        for attempt in range(max_attempts):
            current_llm, current_config, model_name = (state['llm'], state['llm_config'], settings.llm_providers.main_llm.model_name) \
                if attempt == 0 else \
                (state['synthesizer_llm'], state['synthesizer_llm_config'], settings.llm_providers.synthesizer_llm.model_name)

            if attempt > 0:
                logging.warning(f"--- [RETRY] Agent {node_id} failed. Retrying with fallback model: {model_name}. ---")

            agent_timeout = settings.hyperparameters.timeouts.agent_timeout_seconds
            try:
                response_object = await asyncio.wait_for(
                    get_structured_output(
                        llm=current_llm,
                        provider_config=current_config,
                        prompt_template=prompt_template,
                        input_data=prompt_input,
                        pydantic_schema=AgentOutput
                    ),
                    timeout=agent_timeout
                )

                if response_object:
                    logging.info(f"--- AGENT {node_id} SUCCESSFUL OUTPUT (Attempt {attempt+1}) ---")
                    break # Exit loop on success
                else:
                    logging.warning(f"AGENT_NODE_WARNING: Adapter returned None for agent {node_id} on attempt {attempt+1}.")

            except (asyncio.TimeoutError, Exception) as e:
                logging.warning(f"AGENT_NODE_WARNING: Agent {node_id} failed on attempt {attempt + 1}/{max_attempts}. Error: {e}")

        if not response_object:
            error_msg = f"Agent {node_id} failed on all {max_attempts} attempts. Halting graph execution."
            logging.error(error_msg)
            raise AgentExecutionError(message="Failed to produce valid structured output after all retries.", node_id=node_id)
            
        # This code only runs on success
        response_json = response_object.model_dump()
        
        if state.get("is_code_request") and layer_idx > 0:
            logging.info(f"--- [SANDBOX] Testing code from Agent {node_id} ---")
            code_to_test = response_json.get("proposed_solution", "")
            success, output = execute_code_in_sandbox(code_to_test)
            sandbox_log = {
                "sandbox_execution_log": {
                    "success": success,
                    "output": output
                }
            }
            agent_memory.append(sandbox_log)
            logging.info(f"--- [SANDBOX] Agent {node_id} Result: {'Success' if success else 'Failure'} ---")
            logging.info(output)

        current_memory = state.get("memory", {})
        current_memory[node_id] = agent_memory + [response_json]

        return {"agent_outputs": {node_id: response_json}, "memory": current_memory}

    return agent_node