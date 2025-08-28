# app/graph/nodes/reflection_nodes.py
import asyncio
import json
import logging
import random

from app.agents.personas import PersonaService, reactor_list
from app.graph.state import GraphState
from app.services.prompt_service import prompt_service
from app.core.config import settings
from app.services.structured_output_adapter import get_structured_output
from app.services.schemas import (
    AssessmentOutput, NewProblemOutput, DecompositionOutput, AgentAnalysisOutput
)

def create_progress_assessor_node():
    """Creates the node that assesses if significant progress was made."""
    async def assessor_node(state: GraphState) -> dict:
        logging.info("--- [REFLECTION] Assessing Progress ---")
        final_solution = state.get("final_solution")
        llm, llm_config = state["llm"], state["llm_config"]

        if not final_solution or final_solution.get("error"):
            logging.warning("No valid solution to assess. Defaulting to no significant progress.")
            return {"significant_progress_made": False}
        
        try:
            timeout = settings.hyperparameters.timeouts.reflection_timeout_seconds
            assessment = await asyncio.wait_for(
                get_structured_output(
                    llm=llm,
                    provider_config=llm_config,
                    prompt_template=state.get("assessor_prompt", prompt_service.get_template("initial_assessor")),
                    input_data={
                        "original_request": state["original_request"],
                        "proposed_solution": json.dumps(final_solution),
                        "execution_context": ""
                    },
                    pydantic_schema=AssessmentOutput
                ),
                timeout=timeout
            )
            
            if assessment:
                logging.info(f"SUCCESS: Progress assessment complete. Progress: {assessment.significant_progress}. Reasoning: {assessment.reasoning}")
                logging.info(f"ASSESSMENT: Significant Progress = {assessment.significant_progress}. Reason: {assessment.reasoning}")
                return {"significant_progress_made": assessment.significant_progress}
            else:
                raise ValueError("Progress assessor returned no valid output.")
        except Exception as e:
            logging.error(f"ASSESSOR_NODE_ERROR: {e}. Defaulting to no progress.", exc_info=True)
            return {"significant_progress_made": False}
            
    return assessor_node

def create_reframe_and_decompose_node():
    """Creates node to re-frame and decompose the problem after a breakthrough."""
    async def reframe_node(state: GraphState) -> dict:
        logging.info("--- [REFLECTION] Re-framing Problem due to breakthrough ---")
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

def create_update_personas_node(llm):
    """Creates the node that evolves the personas of the reflection agents."""
    async def update_personas_node(state: GraphState) -> dict:
        logging.info("--- [REFLECTION] Updating Personas of Reflection Agents ---")
        agent_utterances = json.dumps(state.get("agent_outputs"))
        selector_chain = prompt_service.create_chain(llm, "pseudo_neurotransmitter_selector")
        
        try:
            selected_formula = await selector_chain.ainvoke({"agent_utterances": agent_utterances})
            selected_formula = selected_formula.strip()
            if not selected_formula:
                selected_formula = random.choice(reactor_list)
                logging.warning(f"Persona selector returned empty. Choosing random: {selected_formula}")

            logging.info(f"Selected new persona formula: {selected_formula}")
            # 2. Map the formula to persona prompts
            mapper = PersonaService()
            reactor_prompts = "\n".join(mapper.table(selected_formula))
            # 3. Create chains to update the prompts for critique and assessor agents
            critique_updater_chain = prompt_service.create_chain(llm, "critique_prompt_updater")
            assessor_updater_chain = prompt_service.create_chain(llm, "assessor_prompt_updater")
            individual_updater_chain = prompt_service.create_chain(llm, "individual_critique_prompt_updater")
            # 4. Asynchronously generate the new prompts
            new_critique_prompt, new_assessor_prompt, new_individual_prompt = await asyncio.gather(
                critique_updater_chain.ainvoke({"reactor_prompts": reactor_prompts}),
                assessor_updater_chain.ainvoke({"reactor_prompts": reactor_prompts}),
                individual_updater_chain.ainvoke({"reactor_prompts": reactor_prompts})
            )
            
            logging.info("Successfully generated new prompts for reflection agents.")
            return {
                "critique_prompt": new_critique_prompt,
                "assessor_prompt": new_assessor_prompt,
                "individual_critique_prompt": new_individual_prompt,
            }
        except Exception as e:
            logging.error(f"UPDATE_PERSONAS_ERROR: Failed to update reflection personas: {e}", exc_info=True)
            return {}

    return update_personas_node

def create_critique_node(llm):
    """Creates the node that generates critiques for the agent's work."""
    async def critique_node(state: GraphState) -> dict:
        logging.info("--- [REFLECTION] Generating Critiques ---")
        
        critiques = {}
        params = state["params"]
        final_solution = state.get("final_solution", {})
        
        if not final_solution:
            logging.warning("No final solution available to critique.")
            return {"critiques": {}}
        
        # 1. Generate Global Critique for the final output
        global_critique_chain = prompt_service.create_chain(llm, "raw_string_template", raw_template_str=state["critique_prompt"])
        global_critique = await global_critique_chain.ainvoke({
            "original_request": state["original_request"],
            "proposed_solution": json.dumps(final_solution)
        })
        critiques["global_critique"] = global_critique
        logging.info(f"Global critique generated: {global_critique[:200]}...")
        
        # 2. Generate Individual Critiques based on strategy
        critique_strategy = params.critique_strategy
        
        agents_to_critique = []
        if critique_strategy == "full":
            # Critique everyone except the final layer
            num_layers_to_critique = len(state['all_layers_prompts']) - 1
            agents_to_critique = [f"agent_{i}_{j}" 
                                  for i in range(num_layers_to_critique)
                                  for j in range(len(state['all_layers_prompts'][i]))]
        elif critique_strategy == "penultimate":
            # Critique only the layer before synthesis
            penultimate_layer_idx = len(state['all_layers_prompts']) - 2
            if penultimate_layer_idx >= 0:
                agents_to_critique = [f"agent_{penultimate_layer_idx}_{j}" 
                                      for j in range(len(state['all_layers_prompts'][penultimate_layer_idx]))]

        async def generate_individual_critique(agent_id: str):
            agent_output = state['agent_outputs'].get(agent_id, {})
            sub_problem = state['decomposed_problems'].get(agent_id, "N/A")
            
            raw_template = state['individual_critique_prompt']
            
            # Now we use the raw string capability of the prompt service
            individual_chain = prompt_service.create_chain(llm, "raw_string_template", raw_template_str=partially_formatted_prompt)

            critique = await individual_chain.ainvoke({
				"agent_id": agent_id,
                "sub_problem": sub_problem,
                "original_request": state['original_request'],
                "final_synthesized_solution": json.dumps(final_solution),
                "agent_output": json.dumps(agent_output),
            })
            critiques[agent_id] = critique
            logging.info(f"Generated critique for {agent_id}")

        await asyncio.gather(*(generate_individual_critique(agent_id) for agent_id in agents_to_critique))
        
        return {"critiques": critiques}
    return critique_node

def create_update_agent_prompts_node():
    """Creates the node that updates agent prompts based on critiques."""
    async def update_prompts_node(state: GraphState) -> dict:
        logging.info("--- [REFLECTION] Updating Agent Prompts (Backpropagation) ---")
        critiques, llm, llm_config, params = state.get("critiques", {}), state["llm"], state["llm_config"], state["params"]
        
        if not critiques and not state.get("significant_progress_made"):
            logging.warning("No critiques and no progress; prompts will not be updated.")
            return {"agent_outputs": {}}

        prompts_copy = [layer[:] for layer in state["all_layers_prompts"]]
        update_failure_count = 0
        timeout = settings.hyperparameters.timeouts.reflection_timeout_seconds
        # The parameters are static for the entire run, so they can be partially formatted once
        dense_spanner = prompt_service.create_chain(llm, "dense_spanner", **params.model_dump())

        for i in range(len(prompts_copy) - 1, -1, -1):
            for j, agent_prompt in enumerate(prompts_copy[i]):
                agent_id = f"agent_{i}_{j}"
                critique = critiques.get(agent_id, critiques.get("global_critique", ""))
                
                if not critique and not state.get("significant_progress_made"): continue
                
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
                    
                    new_prompt = await asyncio.wait_for(dense_spanner.ainvoke({
                        "attributes": analysis.attributes, 
                        "hard_request": analysis.hard_request,
                        "critique": critique, 
                        "sub_problem": sub_problem
                    }), timeout=timeout)
                    prompts_copy[i][j] = new_prompt
                except Exception as e:
                    logging.error(f"PROMPT_UPDATE_ERROR for {agent_id}: {e}. Keeping original prompt.", exc_info=True)
                    update_failure_count += 1
        
        if update_failure_count > 0:
            total_agents = sum(len(layer) for layer in prompts_copy)
            logging.warning(f"{update_failure_count}/{total_agents} agents failed to update prompts.")
        
        logging.info(f"--- Finished Epoch {state.get('epoch', 0)}. ---")
        # Clear out state for the next epoch
        return { "all_layers_prompts": prompts_copy, "agent_outputs": {}, "critiques": {}, "final_solution": None }
    return update_prompts_node