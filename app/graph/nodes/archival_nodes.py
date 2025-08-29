# app/graph/nodes/archival_nodes.py
import asyncio
import json
import re
import traceback
import logging
from langchain_core.documents import Document
from app.graph.state import GraphState
from app.rag.raptor import RAPTOR
from app.services.prompt_service import prompt_service
from app.core.state_manager import session_manager
from app.core.context import ServiceContext

def create_archive_epoch_outputs_node():
    """Creates node to archive agent outputs as documents."""
    async def archive_node(state: GraphState):
        logging.info("--- [ARCHIVAL] Archiving agent outputs for RAG ---")
        epoch_outputs = state.get("agent_outputs", {})
        if not epoch_outputs:
            return {}

        new_docs = [
            Document(
                page_content=f"Agent ID: {agent_id}\nEpoch: {state['epoch']}\nSystem Prompt: {state['all_layers_prompts'][int(agent_id.split('_')[1])][int(agent_id.split('_')[2])]}\nOutput: {json.dumps(output)}",
                metadata={"agent_id": agent_id, "epoch": state['epoch']}
            ) for agent_id, output in epoch_outputs.items()
        ]
        
        all_docs = state.get("all_rag_documents", []) + new_docs
        logging.info(f"Archived {len(new_docs)} docs. Total RAG docs: {len(all_docs)}.")
        return {"all_rag_documents": all_docs}
    return archive_node

async def _update_rag_node_logic(state: GraphState, services: ServiceContext, end_of_run: bool = False):
    node_name = "Final RAG Index" if end_of_run else f"Epoch {state['epoch']} RAG Index"
    logging.info(f"--- [RAG] Building {node_name} ---")

    all_docs = state.get("all_rag_documents", [])
    if not all_docs:
        logging.warning("No documents to index.")
        return {}

    raptor_index = RAPTOR(llm=services.summarizer_llm, embeddings_model=services.embeddings_model)
    try:
        await raptor_index.add_documents(all_docs)
        logging.info(f"SUCCESS: {node_name} built.")

        # This allows the UI to enable the diagnostic chat.
        session_id = state.get('session_id')
        if session_id:
            logging.info(f"Session {session_id} RAG index ready.", extra={
                'ui_extra': {'type': 'session_id', 'data': session_id}
            })
        
        # Update the service context with the new index
        services.raptor_index = raptor_index
        return {}
    except Exception as e:
        logging.error(f"Failed to build {node_name}: {e}\n{traceback.format_exc()}")
        return {}
    
def create_update_rag_index_node():
    """Creates node to build/update the RAPTOR RAG index."""
    async def update_rag_node_wrapper(state: GraphState, config: dict):
        """Wrapper that retrieves services and calls the core logic."""
        session_id = config["configurable"]["session_id"]
        session = session_manager.get_session(session_id)
        if not session:
            raise RuntimeError(f"Session {session_id} not found for RAG index node")
        
        services: ServiceContext = session["services"]
        return await _update_rag_node_logic(state, services, end_of_run=False)

    def update_rag_node_final_wrapper(state: GraphState, config: dict):
        """Special wrapper for the end_of_run=True case."""
        session_id = config["configurable"]["session_id"]
        session = session_manager.get_session(session_id)
        if not session:
            raise RuntimeError(f"Session {session_id} not found for final RAG index node")

        services: ServiceContext = session["services"]
        return asyncio.run(_update_rag_node_logic(state, services, end_of_run=True))
    
    # Return two different functions based on need, a bit of a hack for the '/harvest' endpoint logic.
    # The main graph will use the first one. Harvest will use the second.
    return update_rag_node_wrapper, update_rag_node_final_wrapper

async def _metrics_node_logic(state: GraphState, services: ServiceContext):
    logging.info("--- [METRICS] Calculating Perplexity ---")
    outputs = state.get("agent_outputs", {})
    if not outputs:
        return {}

    combined_text = "\n\n".join(json.dumps(output) for output in outputs.values())
    perplexity_chain = prompt_service.create_chain(services.llm, "perplexity_heuristic")

    try:
        score_str = await perplexity_chain.ainvoke({"text_to_analyze": combined_text})
        score = float(re.sub(r'[^\d.]', '', score_str))
    except (ValueError, TypeError):
        score = 100.0
        logging.error("Could not parse perplexity score. Defaulting to 100.")

    perplexity_data = {'epoch': state['epoch'], 'perplexity': score}
    logging.info(f"Perplexity calculated: {score}", extra={
        'ui_extra': {'type': 'perplexity', 'data': perplexity_data}
    })
    
    new_history = state.get("perplexity_history", []) + [score]
    return {"perplexity_history": new_history, "epoch": state["epoch"] + 1}
    
def create_metrics_node():
    """Creates node for calculating perplexity metrics."""
    async def metrics_node_wrapper(state: GraphState, config: dict) -> dict:
        session_id = config["configurable"]["session_id"]
        session = session_manager.get_session(session_id)
        if not session:
            raise RuntimeError(f"Session {session_id} not found for metrics node")
        
        services: ServiceContext = session["services"]
        return await _metrics_node_logic(state, services)

    return metrics_node_wrapper