# app/graph/workflow.py

import json
import logging
import asyncio 
from langgraph.graph import StateGraph, END
from app.graph.state import GraphState
from app.graph.nodes import (
    create_agent_node, create_synthesis_node, create_archive_epoch_outputs_node,
    create_update_rag_index_node, create_metrics_node, create_progress_assessor_node,
    create_reframe_and_decompose_node, create_critique_node, create_update_personas_node,
    create_update_agent_prompts_node
)
from app.services.agent_factory import AgentFactory
from app.services.prompt_service import prompt_service
from app.services.structured_output_adapter import get_structured_output
from app.services.schemas import DecompositionOutput
from app.api.schemas import GraphRunParams
from app.core.config import settings

async def build_graph_workflow(params: GraphRunParams, llm, llm_config, synthesizer_llm, summarizer_llm, embeddings_model):
    """
    Builds the complete LangGraph workflow by defining nodes and their connections.
    """
    user_prompt = params.prompt
    num_agents_per_layer = len(params.mbti_archetypes)
    cot_trace_depth = params.cot_trace_depth
    total_agents = num_agents_per_layer * cot_trace_depth

    logging.info("--- Decomposing Original Problem into Subproblems ---")
    try:
        # Use the structured output adapter directly for decomposition.
        timeout = settings.hyperparameters.timeouts.reflection_timeout_seconds
        decomposition_result = await asyncio.wait_for(
            get_structured_output(
                llm=llm,
                provider_config=llm_config,
                prompt_template=prompt_service.get_template("problem_decomposition"),
                input_data={"problem": user_prompt, "num_sub_problems": total_agents},
                pydantic_schema=DecompositionOutput
            ),
            timeout=timeout
        )
        if not decomposition_result or not decomposition_result.sub_problems or len(decomposition_result.sub_problems) != total_agents:
            raise ValueError(f"Problem decomposition failed or produced incorrect number of sub-problems. Expected {total_agents}, got {len(decomposition_result.sub_problems) if decomposition_result else 0}.")

        # Create the map of sub-problems to agent IDs.
        decomposed_problems_map = {
            f"agent_{i}_{j}": decomposition_result.sub_problems[i * num_agents_per_layer + j]
            for i in range(cot_trace_depth)
            for j in range(num_agents_per_layer)
        }
        logging.info(f"SUCCESS: Decomposed into {len(decomposed_problems_map)} subproblems.")

        # Now, instantiate and use the AgentFactory to create the prompts.
        agent_factory = AgentFactory(llm=llm, llm_config=llm_config)
        all_layers_prompts = await agent_factory.create_all_agent_prompts(
            decomposed_problems_map=decomposed_problems_map,
            params=params
        )
        logging.info("SUCCESS: Generated all agent prompts.")

    except Exception as e:
        logging.critical(f"FATAL: Agent prompt generation or problem decomposition failed. Graph setup cannot continue. Error: {e}", exc_info=True)
        raise e
    
    workflow = StateGraph(GraphState)

    def start_epoch(state: GraphState):
        current_epoch_display = state['epoch']
        logging.info(f"==================== STARTING EPOCH {current_epoch_display + 1} / {state['max_epochs']} ====================")
        return {}
    workflow.add_node("start_epoch", start_epoch)
    workflow.set_entry_point("start_epoch")

    for i, layer_prompts in enumerate(all_layers_prompts):
        for j, _ in enumerate(layer_prompts):
            node_id = f"agent_{i}_{j}"
            workflow.add_node(node_id, create_agent_node(node_id))
    
    workflow.add_node("synthesis", create_synthesis_node())
    workflow.add_node("archive_epoch_outputs", create_archive_epoch_outputs_node())
    update_rag_node_func = create_update_rag_index_node(summarizer_llm, embeddings_model)
    workflow.add_node("update_rag_index", update_rag_node_func)
    workflow.add_node("metrics", create_metrics_node(llm))
    workflow.add_node("update_personas", create_update_personas_node(llm))
    workflow.add_node("progress_assessor", create_progress_assessor_node())
    workflow.add_node("reframe_and_decompose", create_reframe_and_decompose_node())
    workflow.add_node("update_prompts", create_update_agent_prompts_node())
    workflow.add_node("critique", create_critique_node(llm))
    workflow.add_node("build_final_rag_index", lambda state: update_rag_node_func(state, end_of_run=True))
    
    logging.info("--- Connecting Graph Nodes ---")

    layer_node_ids = [[f"agent_{i}_{j}" for j in range(num_agents_per_layer)] for i in range(cot_trace_depth)]

    for node in layer_node_ids[0]:
        workflow.add_edge("start_epoch", node)

    for i in range(cot_trace_depth - 1):
        for end_node in layer_node_ids[i + 1]:
            workflow.add_edge(layer_node_ids[i], end_node)

    workflow.add_edge(layer_node_ids[-1], "synthesis")
    workflow.add_edge("synthesis", "archive_epoch_outputs")
    workflow.add_edge("archive_epoch_outputs", "update_rag_index")
    workflow.add_edge("update_rag_index", "metrics")
    
    workflow.add_conditional_edges(
        "metrics",
        lambda state: "build_final_rag_index" if state["epoch"] >= state["max_epochs"] else "progress_assessor",
        {"build_final_rag_index": END, "progress_assessor": "progress_assessor"}
    )
    workflow.add_conditional_edges(
        "progress_assessor",
        lambda state: "reframe_and_decompose" if state["significant_progress_made"] else "update_personas",
        {"reframe_and_decompose": "reframe_and_decompose", "update_personas": "update_personas"}
    )

    workflow.add_edge("update_personas", "critique")
    workflow.add_edge("critique", "update_prompts")
    workflow.add_edge("reframe_and_decompose", "update_prompts")
    workflow.add_edge("update_prompts", "start_epoch")

    graph = workflow.compile()
    
    # This special message for the UI sent using the logging `extra` parameter.
    logging.info("Graph compiled. Visualization ready.", extra={
        'ui_extra': {
            'type': 'graph',
            'data': graph.get_graph().draw_ascii()
        }
    })
    
    initial_state_components = {
        "decomposed_problems": decomposed_problems_map,
        "all_layers_prompts": all_layers_prompts
    }
    return graph, initial_state_components