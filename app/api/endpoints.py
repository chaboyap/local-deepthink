# app/api/endpoints.py

import asyncio
import io
import json
import re
import traceback
import zipfile
import logging
from typing import Dict, Any
from fastapi import APIRouter, Request, Body, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from sse_starlette.sse import EventSourceResponse
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

from app.core.config import settings
from app.core.state_manager import session_manager
from app.graph.nodes import (
    create_final_harvest_node, create_update_rag_index_node,
    create_synthesis_node, create_code_execution_node, create_agent_node
)
from app.graph.state import GraphState
from app.graph.workflow import build_graph_workflow
from app.services.llm_service import initialize_llms, LLMInitializationParams
from app.services.prompt_service import prompt_service
from app.api.schemas import GraphRunPayload, GraphRunParams
from app.utils.exceptions import AgentExecutionError
from app.utils.mock_llms import CoderMockLLM, MockLLM
from app.utils.json_utils import clean_and_parse_json


router = APIRouter()

@router.get("/config")
async def get_app_config():
    """
    Provides essential configuration details to the frontend,
    including the main model name and default hyperparameters.
    """
    try:
        hyperparams = settings.hyperparameters
        frontend_defaults = {
            "cot_trace_depth": hyperparams.cot_trace_depth,
            "num_epochs": hyperparams.num_epochs,
            "num_questions": hyperparams.num_questions_for_harvest,
            "vector_word_size": hyperparams.vector_word_size,
            "prompt_alignment": hyperparams.prompt_alignment,
            "density": hyperparams.density,
            "prompt": hyperparams.default_prompt,
            "mbti_archetypes": hyperparams.default_mbti_selection
        }
        
        return JSONResponse(content={
            "default_model": settings.llm_providers.main_llm.model_name,
            "defaults": frontend_defaults
        })
    except Exception as e:
        logging.error(f"Failed to load configuration for /config endpoint: {e}", exc_info=True)
        return JSONResponse(
            content={"error": "Could not load configuration.", "details": str(e)}, 
            status_code=500
        )

@router.post("/build_and_run_graph")
async def build_and_run_graph(payload: GraphRunPayload):
    """Endpoint to build and execute the agent graph based on user parameters."""
    user_prompt = payload.params.prompt
    logging.info(f"--- Starting graph run with params: {payload.params.model_dump_json(indent=2)} ---")
 
    try:
        init_params = LLMInitializationParams(
            debug_mode=payload.params.debug_mode,
            coder_debug_mode=payload.params.coder_debug_mode
        )
        (llm, llm_config), \
        (synthesizer_llm, synthesizer_llm_config), \
        (summarizer_llm, summarizer_llm_config), \
        (embeddings_model, embeddings_config) = await initialize_llms(init_params)
        
        logging.info("--- [SETUP] Initialized LLMs successfully.")
        
        request_is_code_chain = prompt_service.create_chain(llm, "request_is_code")
        
        is_code_str = (await request_is_code_chain.ainvoke({"request": user_prompt})).lower().strip()
        is_code = "true" in is_code_str

        if payload.params.coder_debug_mode: is_code = True
        elif payload.params.debug_mode: is_code = False

        logging.info(f"--- [SETUP] Request is code: {is_code}")

        logging.info("--- [SETUP] Attempting to build graph workflow...")
        graph, initial_state_components = await build_graph_workflow(
            payload.params, llm, llm_config, synthesizer_llm, summarizer_llm, embeddings_model, is_code
        )
        logging.info("--- [SETUP] Graph workflow built successfully.")
        
        initial_state = {
            "previous_solution": "",
            "chat_history": [],
            "original_request": user_prompt,
            "current_problem": user_prompt,
            "epoch": 0,
            "max_epochs": payload.params.num_epochs,
            "params": payload.params,
            "agent_outputs": {},
            "memory": {},
            "final_solution": None,
            "perplexity_history": [],
            "raptor_index": None,
            "all_rag_documents": [],
            "academic_papers": None,
            "is_code_request": is_code,
            "llm": llm,
            "llm_config": llm_config,
            "summarizer_llm": summarizer_llm,
            "summarizer_llm_config": summarizer_llm_config,
            "embeddings_model": embeddings_model,
            "embeddings_config": embeddings_config,
            "synthesizer_llm": synthesizer_llm,
            "synthesizer_llm_config": synthesizer_llm_config,
            "modules": [],
            "synthesis_context_queue": [],
            "synthesis_execution_success": True,
            **initial_state_components
        }
        
        loggable_state = {k: (f"<{v.__class__.__name__}>" if hasattr(v, '__class__') and 'langchain' in str(v.__class__) else v) for k, v in initial_state.items()}
        logging.info(f"--- Graph Initial State ---\n{json.dumps(loggable_state, indent=2, default=str)}")
        
        session_id = session_manager.create_session(initial_state)
        
        logging.info(f"--- Starting Execution (Epochs: {payload.params.num_epochs}) ---")

        try:
            recursion_limit = (payload.params.num_epochs * len(graph.nodes)) + 50
            initial_session_object = session_manager.get_session(session_id)
            if not initial_session_object:
                raise RuntimeError("Failed to create and retrieve session immediately.")

            async for output in graph.astream(initial_session_object['state'], {'recursion_limit': recursion_limit}):
                for node_name, node_output in output.items():
                    logging.info(f"--- Node Finished: {node_name} ---")
                    if isinstance(node_output, dict) and node_output:
                        session_manager.update_session_state(session_id, node_output)
        
        except AgentExecutionError as e:
            error_message = f"Execution halted due to a critical failure in agent '{e.node_id}'. Reason: {e.message}"
            logging.error("--- FATAL EXECUTION ERROR ---")
            logging.error(error_message)
            return JSONResponse(content={"message": error_message}, status_code=500)
        
        final_session_object = session_manager.get_session(session_id)
        if not final_session_object:
            logging.error(f"Session {session_id} expired or was lost before completion.")
            return JSONResponse(content={"message": f"Session {session_id} could not be found after execution."}, status_code=404)
        
        final_state_value = final_session_object['state']
        session_manager.set_final_state(session_id, final_state_value)

        if is_code:
            final_code_solution = final_state_value.get("final_solution", {})
            final_modules = final_state_value.get("modules", [])
            logging.info(f"--- 💻 Code Generation Finished. Returning final code and {len(final_modules)} modules. ---")
            return JSONResponse(content={
                "message": "Code generation complete.",
                "code_solution": final_code_solution.get("proposed_solution", "# No code generated."),
                "reasoning": final_code_solution.get("reasoning", "No reasoning provided."),
                "modules": final_modules,
                "session_id": session_id
            })
        else:
            logging.info("--- Agent Execution Finished. Pausing for User Chat. ---")
            return JSONResponse(content={"message": "Chat is now active.", "session_id": session_id})

    except Exception as e:
        error_message = f"An unexpected error occurred during graph setup: {e}"
        tb_str = traceback.format_exc()
        logging.critical(f"--- FATAL ERROR DURING GRAPH SETUP ---", exc_info=True)
        return JSONResponse(content={"message": error_message, "traceback": tb_str}, status_code=500)

@router.post("/run_inference_from_state")
async def run_inference_from_state(payload: dict = Body(...)):
    """Runs inference on an imported QNN state without further training."""
    logging.info("--- [INFERENCE-ONLY] Received request to run inference from imported state. ---")
    try:
        imported_state = payload.get("imported_state")
        user_prompt = payload.get("prompt")
        params_dict = imported_state.get("params", {})

        if not imported_state or not user_prompt:
            return JSONResponse(content={"error": "Invalid payload. 'imported_state' and 'prompt' are required."}, status_code=400)

        # Re-initialize LLM for the inference run
        params = GraphRunParams(**params_dict)
        init_params = LLMInitializationParams(
            debug_mode=params.debug_mode,
            coder_debug_mode=params.coder_debug_mode
        )
        (llm, _), (synthesizer_llm, _), _, _ = await initialize_llms(init_params)
        
        all_layers_prompts = imported_state.get("all_layers_prompts", [])
        if not all_layers_prompts:
            return JSONResponse(content={"error": "Imported state must contain 'all_layers_prompts'."}, status_code=400)

        inference_state = imported_state.copy()
        inference_state.update({
            "original_request": user_prompt,
            "current_problem": user_prompt,
            "agent_outputs": {},
            "synthesizer_llm": synthesizer_llm,
            "llm": llm,
            "is_code_request": True, # Assume inference is for code, can be improved
            "synthesis_context_queue": imported_state.get("synthesis_context_queue", [])
        })

        workflow = StateGraph(GraphState)

        # Add agent nodes
        for i, layer in enumerate(all_layers_prompts):
            for j in range(len(layer)):
                node_id = f"agent_{i}_{j}"
                workflow.add_node(node_id, create_agent_node(node_id))

        # Add synthesis node
        workflow.add_node("synthesis", create_synthesis_node())

        # Define layer node IDs for clarity
        layer_node_ids = [[f"agent_{i}_{j}" for j in range(len(all_layers_prompts[i]))] for i in range(len(all_layers_prompts))]
        
        # Set entry point for parallel execution of the first layer
        workflow.set_entry_point(layer_node_ids[0])
        
        # Connect subsequent layers correctly for parallel execution
        for i in range(len(layer_node_ids) - 1):
            workflow.add_edge(layer_node_ids[i], layer_node_ids[i+1])
        
        # Connect last layer to synthesis
        workflow.add_edge(layer_node_ids[-1], "synthesis")
        workflow.add_edge("synthesis", END)

        graph = workflow.compile()
        logging.info("Inference graph compiled.", extra={'ui_extra': {'type': 'graph', 'data': graph.get_graph().draw_ascii()}})
        
        final_result_node = None
        async for output in graph.astream(inference_state):
             if "synthesis" in output:
                final_result_node = output["synthesis"]

        logging.info("--- [INFERENCE-ONLY] Run complete. ---")

        if not final_result_node:
            logging.error("--- [INFERENCE-ONLY] Run failed before reaching synthesis node. ---")
            return JSONResponse(content={"message": "Inference run failed before a solution could be synthesized."}, status_code=500)

        synthesis_output = final_result_node.get("final_solution", {})
        return JSONResponse(content={
            "message": "Inference complete.",
            "code_solution": synthesis_output.get("proposed_solution", "No solution generated."),
            "reasoning": synthesis_output.get("reasoning", "No reasoning provided."),
            "is_inference": True
        })
    except Exception as e:
        error_message = f"An error occurred during inference: {e}"
        logging.error(error_message, exc_info=True)
        return JSONResponse(content={"message": error_message, "traceback": traceback.format_exc()}, status_code=500)

@router.get("/export_qnn/{session_id}")
async def export_qnn(session_id: str):
    """Exports the current state of a session graph to a JSON file."""
    final_state = session_manager.get_final_state(session_id)
    if not final_state:
        session_object = session_manager.get_session(session_id)
        if not session_object:
            return JSONResponse(content={"error": "Session not found."}, status_code=404)
        final_state = session_object.get('state')

    if not final_state:
         return JSONResponse(content={"error": "Session state is empty."}, status_code=404)
    
    state_to_export = final_state.copy()
    
    # Remove non-serializable objects
    keys_to_remove = [
        'llm', 'synthesizer_llm', 'summarizer_llm', 'embeddings_model', 
        'raptor_index', 'llm_config', 'synthesizer_llm_config', 
        'summarizer_llm_config', 'embeddings_config'
    ]
    for key in keys_to_remove:
        state_to_export.pop(key, None)
    
    # Serialize Pydantic and Document objects
    if 'params' in state_to_export and isinstance(state_to_export['params'], GraphRunParams):
        state_to_export['params'] = state_to_export['params'].model_dump()

    if 'all_rag_documents' in state_to_export:
        state_to_export['all_rag_documents'] = [doc.dict() for doc in state_to_export['all_rag_documents']]

    logging.info(f"--- [EXPORT] Exporting QNN for session {session_id} ---")
    return JSONResponse(
        content=state_to_export,
        headers={"Content-Disposition": f"attachment; filename=qnn_state_{session_id}.json"}
    )

@router.post("/import_qnn")
async def import_qnn(file: UploadFile = File(...)):
    """Imports a QNN JSON file to initialize a new session."""
    try:
        content = await file.read()
        imported_data = json.loads(content)
        
        # Deserialize Document objects
        if 'all_rag_documents' in imported_data:
            imported_data['all_rag_documents'] = [Document(**doc) for doc in imported_data['all_rag_documents']]
        
        # Re-hydrate and validate Pydantic model
        if 'params' in imported_data:
            imported_data['params'] = GraphRunParams(**imported_data['params'])

        session_id = session_manager.create_session(imported_data)
        logging.info(f"--- [IMPORT] Successfully imported QNN file. New Session ID: {session_id} ---")
        
        return JSONResponse(content={
            "message": "QNN file imported successfully.",
            "session_id": session_id,
            "imported_params": imported_data.get("params", {}).model_dump() if 'params' in imported_data else {}
        })
    except Exception as e:
        error_message = f"Failed to import QNN file: {e}"
        logging.error(error_message, exc_info=True)
        return JSONResponse(content={"message": error_message, "traceback": traceback.format_exc()}, status_code=500)


@router.post("/chat")
async def chat_with_index(payload: dict = Body(...)):
    session_id = payload.get("session_id")
    session_object = session_manager.get_session(session_id)
    if not session_object:
        return JSONResponse(content={"error": "Invalid session ID"}, status_code=404)

    state = session_object["state"]
    raptor_index = state.get("raptor_index")
    llm = state.get("llm")
    
    if not raptor_index or not llm:
        return JSONResponse(content={"error": "RAG index or LLM not found"}, status_code=500)

    async def stream_response():
        message = payload.get("message")
        
        # Lock immediately to append the user's message to the history
        async with session_object["lock"]:
            session_object["state"]["chat_history"].append({"role": "user", "content": message})

        try:
            retrieved_docs = await asyncio.to_thread(raptor_index.retrieve, message, k=10)
            context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
            chat_chain = prompt_service.create_chain(llm, "rag_chat", is_chat_prompt=True)
            
            full_response = ""
            async for chunk in chat_chain.astream({"context": context, "question": message}):
                content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                yield content
                full_response += content
            
            # Lock again to append the final AI response
            async with session_object["lock"]:
                current_state = session_object["state"]
                # Ensure the last entry is the user message to prevent double-adding AI response
                if current_state["chat_history"] and current_state["chat_history"][-1]['role'] == 'user':
                    current_state["chat_history"].append({"role": "ai", "content": full_response})
                
        except Exception as e:
            logging.error(f"Error during chat stream for session {session_id}: {e}", exc_info=True)
            yield f"Error: Could not generate response. {e}"
    return StreamingResponse(stream_response(), media_type="text/event-stream")

@router.post("/diagnostic_chat")
async def diagnostic_chat_with_index(payload: dict = Body(...)):
    session_id = payload.get("session_id")
    session_object = session_manager.get_session(session_id)
    if not session_object:
        return JSONResponse(content={"error": "Invalid session ID"}, status_code=404)

    state = session_object["state"]
    raptor_index = state.get("raptor_index")
    if not raptor_index:
        return StreamingResponse((_ for _ in ["The RAG index is not yet available."]), media_type="text/event-stream")
        
    async def stream_response():
        query = payload.get("message", "").strip()
        try:
            logging.info(f"--- [DIAGNOSTIC] Raw RAG query received: '{query}' ---")
            retrieved_docs = await asyncio.to_thread(raptor_index.retrieve, query, k=10)
            if not retrieved_docs:
                yield "No relevant documents found for that query."
                return
            yield "--- Top Relevant Documents (Raw Retrieval) ---\n\n"
            for i, doc in enumerate(retrieved_docs):
                yield (f"DOCUMENT #{i+1}\n-----------------\n"
                       f"METADATA: {json.dumps(doc.metadata)}\n"
                       f"CONTENT: {doc.page_content.replace('\\n', ' ').strip()}...\n\n")
        except Exception as e:
            logging.error(f"Error during diagnostic chat stream for session {session_id}: {e}", exc_info=True)
            yield f"Error: Could not generate response. {e}"
    return StreamingResponse(stream_response(), media_type="text/event-stream")

@router.post("/harvest")
async def harvest_session(payload: dict = Body(...)):
    session_id = payload.get("session_id")
    session_object = session_manager.get_session(session_id)
    if not session_object:
        return JSONResponse(content={"error": "Invalid session ID"}, status_code=404)
    
    async with session_object["lock"]:
        try:
            state = session_object["state"]
            logging.info("--- [HARVEST] Initiating Final Harvest Process ---")
            chat_history = state.get("chat_history", [])
            
            if chat_history:
                chat_docs = [Document(page_content=f"User: {chat_history[i-1]['content']}\nAI: {turn['content']}", metadata={"source": "chat", "turn": i//2})
                             for i, turn in enumerate(chat_history) if turn['role'] == 'ai']
                
                state["all_rag_documents"].extend(chat_docs)
                session_manager.update_session_state(session_id, {"all_rag_documents": state["all_rag_documents"]})

            # Create a mini-graph for the harvest process to run asynchronously
            harvest_workflow = StateGraph(GraphState)
            
            update_rag_node = create_update_rag_index_node(state["summarizer_llm"], state["embeddings_model"])
            harvest_workflow.add_node("update_rag_index_final", lambda s: update_rag_node(s, end_of_run=True))

            num_questions = int(state["params"].num_questions_for_harvest)
            harvest_node = create_final_harvest_node(state["llm"], state["synthesizer_llm"], num_questions)
            harvest_workflow.add_node("final_harvest", harvest_node)

            harvest_workflow.set_entry_point("update_rag_index_final")
            harvest_workflow.add_edge("update_rag_index_final", "final_harvest")
            harvest_workflow.add_edge("final_harvest", END)
            
            harvest_graph = harvest_workflow.compile()
            
            logging.info("--- [HARVEST] Executing Harvest Graph... ---")
            async for output in harvest_graph.astream(state):
                for node_name, node_output in output.items():
                    logging.info(f"--- [HARVEST] Node Finished: {node_name} ---")
                    if isinstance(node_output, dict) and node_output:
                        session_manager.update_session_state(session_id, node_output)
            
            updated_session_object = session_manager.get_session(session_id)
            academic_papers = updated_session_object["state"].get("academic_papers", {})

            if academic_papers:
                session_manager.store_report(session_id, academic_papers)
                logging.info(f"SUCCESS: Final report with {len(academic_papers)} papers created for session {session_id}.")
            
            return JSONResponse(content={"message": "Harvest complete."})
        
        except Exception as e:
            error_message = f"An error occurred during harvest: {e}"
            logging.error(f"--- FATAL ERROR DURING HARVEST ---", exc_info=True)
            return JSONResponse(content={"message": error_message, "traceback": traceback.format_exc()}, status_code=500)

@router.get('/stream_log')
async def stream_log(request: Request):
    """
    Streams log messages to the client using Server-Sent Events.
    This endpoint simply retrieves pre-formatted JSON strings from the
    log queue and yields them to the client. The UIJSONFormatter is responsible
    for creating the structured payloads.
    """
    async def event_generator():
        while True:
            if await request.is_disconnected():
                logging.info("Client disconnected from log stream.")
                break
            try:
                log_json_string = await asyncio.wait_for(session_manager.log_stream.get(), timeout=1.0)
                yield {"data": log_json_string}

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logging.error(f"Error in log stream: {e}")
                break

    return EventSourceResponse(event_generator())

@router.get("/download_report/{session_id}")
async def download_report(session_id: str):
    papers = session_manager.get_report(session_id)
    if not papers:
        return JSONResponse(content={"error": "Report not found or expired."}, status_code=404)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for i, (question, content) in enumerate(papers.items()):
            safe_question = re.sub(r'[^\w\s-]', '', question).strip().replace(' ', '_')
            filename = f"paper_{i+1}_{safe_question[:50]}.md"
            zip_file.writestr(filename, content)
    zip_buffer.seek(0)
    return StreamingResponse(zip_buffer, media_type="application/zip", headers={"Content-Disposition": f"attachment; filename=NOA_Report_{session_id}.zip"})