# app/graph/nodes/reflection_nodes.py
import asyncio
import json
import logging
import random

from app.graph.state import GraphState
from app.services.prompt_service import prompt_service
from app.core.config import settings
from app.services.structured_output_adapter import get_structured_output
from app.services.schemas import (
    NewProblemOutput, DecompositionOutput, AgentAnalysisOutput
)

def create_reframe_and_decompose_node():
    """Creates node to re-frame and decompose the problem after an epoch."""
    async def reframe_node(state: GraphState) -> dict:
        logging.info("--- [REFLECTION] Re-framing Problem for Next Epoch ---")
        llm, llm_config = state["llm"], state["llm_config"]
        timeout = settings.hyperparameters.timeouts.reflection_timeout_seconds
        
        try:
            new_problem_obj = await asyncio.wait_for(
                get_structured_output(
                    llm=llm,
                    provider_config=llm_config,
                    prompt_template=prompt_service.get_template("problem_reframer"),
                    input_data={
                        "original_request": state["original_request"],
                        "final_solution": json.dumps(state.get("final_solution")),
                        "current_problem": state.get("current_problem")
                    },
                    pydantic_schema=NewProblemOutput
                ),
                timeout=timeout
            )
            if not new_problem_obj: raise ValueError("Re-framer returned no valid output.")
            new_problem = new_problem_obj.new_problem
            logging.info(f"SUCCESS: New problem framed: {new_problem}")

            num_agents = sum(len(layer) for layer in state["all_layers_prompts"])
            
            decomp_obj = await asyncio.wait_for(
                get_structured_output(
                    llm=llm,
                    provider_config=llm_config,
                    prompt_template=prompt_service.get_template("problem_decomposition"),
                    input_data={"problem": new_problem, "num_sub_problems": num_agents},
                    pydantic_schema=DecompositionOutput
                ),
                timeout=timeout
            )
            if not decomp_obj or len(decomp_obj.sub_problems) != num_agents:
                 raise ValueError(f"Decomposition of new problem failed. Expected {num_agents} problems, got {len(decomp_obj.sub_problems) if decomp_obj else 0}.")

            new_map = {f"agent_{i}_{j}": decomp_obj.sub_problems[i * len(state['all_layers_prompts'][i]) + j]
                       for i in range(len(state['all_layers_prompts'])) for j in range(len(state['all_layers_prompts'][i]))}

            return {"decomposed_problems": new_map, 'current_problem': new_problem}
        except Exception as e:
            logging.error(f"REFRAME_NODE_ERROR: {e}. Problem will not be updated.", exc_info=True)
            return {}
    return reframe_node

def create_update_agent_prompts_node():
    """Creates the node that updates agent prompts based on the newly reframed problem."""
    async def update_prompts_node(state: GraphState) -> dict:
        logging.info("--- [REFLECTION] Updating Agent Prompts (Targeted Backpropagation) ---")
        llm, llm_config, params = state["llm"], state["llm_config"], state["params"]
        
        prompts_copy = [layer[:] for layer in state["all_layers_prompts"]]
        update_failure_count = 0
        timeout = settings.hyperparameters.timeouts.reflection_timeout_seconds
        
        # Partially format the chain with static parameters for efficiency
        dense_spanner_chain = prompt_service.create_chain(llm, "dense_spanner", **params.model_dump())

        for i in range(len(prompts_copy) - 1, -1, -1):
            for j, agent_prompt in enumerate(prompts_copy[i]):
                agent_id = f"agent_{i}_{j}"
                
                try:
                    analysis = await asyncio.wait_for(
                        get_structured_output(
                            llm=llm,
                            provider_config=llm_config,
                            prompt_template=prompt_service.get_template("attribute_and_hard_request_generator"),
                            input_data={"agent_prompt": agent_prompt, "vector_word_size": params.vector_word_size},
                            pydantic_schema=AgentAnalysisOutput
                        ), timeout=timeout
                    )
                    if not analysis: raise ValueError("Agent analysis returned no output.")
                    
                    sub_problem = state.get("decomposed_problems", {}).get(agent_id, state["original_request"])
                    agent_persona = state.get("agent_personas", {}).get(agent_id, {})
                    
                    new_prompt = await asyncio.wait_for(dense_spanner_chain.ainvoke({
                        "attributes": analysis.attributes, 
                        "hard_request": analysis.hard_request,
                        "sub_problem": sub_problem,
                        "mbti_type": agent_persona.get("mbti_type"),
                        "name": agent_persona.get("name")
                    }), timeout=timeout)
                    prompts_copy[i][j] = new_prompt
                except Exception as e:
                    logging.error(f"PROMPT_UPDATE_ERROR for {agent_id}: {e}. Keeping original prompt.", exc_info=True)
                    update_failure_count += 1
        
        if update_failure_count > 0:
            total_agents = sum(len(layer) for layer in prompts_copy)
            logging.warning(f"{update_failure_count}/{total_agents} agents failed to update prompts.")
        
        logging.info(f"--- Finished Epoch {state.get('epoch', 0)}. ---")
        # Clear out state for the next epoch, carrying over the solution from this one
        previous_solution_str = json.dumps(state.get("final_solution")) if state.get("final_solution") else ""
        return { 
            "all_layers_prompts": prompts_copy, 
            "agent_outputs": {}, 
            "previous_solution": previous_solution_str,
            "final_solution": None
        }
    return update_prompts_node