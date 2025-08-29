# app/graph/workflow.py

import json
import logging
import asyncio 
from langgraph.graph import StateGraph, END
from app.graph.state import GraphState
from app.graph.nodes import (
    create_agent_node, create_synthesis_node, create_archive_epoch_outputs_node,
    create_update_rag_index_node, create_metrics_node,
    create_reframe_and_decompose_node,
    create_update_agent_prompts_node, create_code_execution_node
)
from app.services.agent_factory import AgentFactory
from app.services.prompt_service import prompt_service
from app.services.structured_output_adapter import get_structured_output
from app.services.schemas import DecompositionOutput
from app.api.schemas import GraphRunParams
from app.core.config import settings

async def build_graph_workflow(params: GraphRunParams, llm, llm_config, synthesizer_llm, summarizer_llm, embeddings_model, is_code: bool):
    """
    Builds the complete LangGraph workflow by defining nodes and their connections.
    """
    user_prompt = params.prompt
    num_agents_per_layer = len(params.mbti_archetypes)
    cot_trace_depth = params.cot_trace_depth
    total_agents = num_agents_per_layer * cot_trace_depth

    logging.info("--- Decomposing Original Problem into Subproblems ---")
    try:
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

        decomposed_problems_map = {
            f"agent_{i}_{j}": decomposition_result.sub_problems[i * num_agents_per_layer + j]
            for i in range(cot_trace_depth)
            for j in range(num_agents_per_layer)
        }
        logging.info(f"SUCCESS: Decomposed into {len(decomposed_problems_map)} subproblems.")

        agent_factory = AgentFactory(llm=llm, llm_config=llm_config)
        all_layers_prompts, agent_personas = await agent_factory.create_all_agent_prompts(
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
    if is_code: workflow.add_node("code_execution", create_code_execution_node())
    workflow.add_node("archive_epoch_outputs", create_archive_epoch_outputs_node())
    update_rag_node_func = create_update_rag_index_node(summarizer_llm, embeddings_model)
    workflow.add_node("update_rag_index", update_rag_node_func)
    workflow.add_node("metrics", create_metrics_node(llm))
    workflow.add_node("reframe_and_decompose", create_reframe_and_decompose_node())
    workflow.add_node("update_prompts", create_update_agent_prompts_node())
    
    logging.info("--- Connecting Graph Nodes ---")

    layer_node_ids = [[f"agent_{i}_{j}" for j in range(num_agents_per_layer)] for i in range(cot_trace_depth)]

    # Start all nodes in the first layer in parallel
    workflow.add_edge("start_epoch", layer_node_ids[0])

    # Connect subsequent layers
    for i in range(cot_trace_depth - 1):
        workflow.add_edge(layer_node_ids[i], layer_node_ids[i+1])

    workflow.add_edge(layer_node_ids[-1], "synthesis")

    if is_code:
        workflow.add_edge("synthesis", "code_execution")
        workflow.add_edge("code_execution", "archive_epoch_outputs")
    else:
        workflow.add_edge("synthesis", "archive_epoch_outputs")

    workflow.add_edge("archive_epoch_outputs", "update_rag_index")
    workflow.add_edge("update_rag_index", "metrics")
    
    def assess_progress_and_decide_path(state: GraphState):
        # First, check for code execution failure, which is a reason to loop again
        if state.get("is_code_request") and not state.get("synthesis_execution_success", True):
            logging.warning("Code execution failed. Attempting another epoch.")
            if state["epoch"] >= state["max_epochs"]:
                logging.error(f"Final epoch ({state['epoch']}) finished after code failure. Ending run.")
                return END
            else:
                return "reframe_and_decompose"
        
        # If code execution succeeded OR it's not a code request, use the standard epoch check
        if state["epoch"] >= state["max_epochs"]:
            logging.info(f"LOG: Final epoch ({state['epoch']}) finished. Proceeding to build final report.")
            # For non-code tasks, there's no chat, so we end. For code-tasks that succeed on the last epoch, we also end.
            return END
        else:
            return "reframe_and_decompose"

    workflow.add_conditional_edges(
        "metrics",
        assess_progress_and_decide_path,
        {
            "reframe_and_decompose": "reframe_and_decompose",
            END: END
        }
    )

    workflow.add_edge("reframe_and_decompose", "update_prompts")
    workflow.add_edge("update_prompts", "start_epoch")

    graph = workflow.compile()
    
    logging.info("Graph compiled. Visualization ready.", extra={
        'ui_extra': { 'type': 'graph', 'data': graph.get_graph().draw_ascii() }
    })
    
    initial_state_components = {
        "decomposed_problems": decomposed_problems_map,
        "all_layers_prompts": all_layers_prompts,
        "agent_personas": agent_personas
    }
    return graph, initial_state_components