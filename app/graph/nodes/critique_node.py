# app/graph/nodes/critique_node.py

import asyncio
import json
import logging
from app.graph.state import GraphState
from app.services.prompt_service import prompt_service
from app.core.config import CritiqueAgentConfig

def create_critique_node(critique_config: CritiqueAgentConfig):
    """Factory for creating a critique node based on a persona from the config."""
    async def critique_node(state: GraphState) -> dict:
        node_name = f"CritiqueAgent_{critique_config.name.replace(' ', '_')}"
        logging.info(f"--- [CRITIQUE] Invoking {node_name} ---")

        synthesized_solution = state.get("final_solution", {})
        if not synthesized_solution:
            logging.warning(f"{node_name}: No solution to critique. Skipping.")
            return {}

        critique_prompt_template = prompt_service.get_template("critique")
        
        # Partially format the prompt with the persona before creating the chain.
        # This is handled by the "raw_string_template" key in prompt_service.
        critique_chain = prompt_service.create_chain(
            state["llm"], 
            "raw_string_template",
            raw_template_str=critique_prompt_template.format(persona=critique_config.persona)
        )
        
        critique_text = await critique_chain.ainvoke({
            "original_request": state["original_request"],
            "proposed_solution": json.dumps(synthesized_solution, indent=2)
        })

        logging.info(f"--- {node_name} Output ---\n{critique_text}")
        
        return {"critiques": {critique_config.name: critique_text}}

    return critique_node