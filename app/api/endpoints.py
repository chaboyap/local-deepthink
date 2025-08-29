# app/api/endpoints.py

import asyncio
import io
import json
import re
import traceback
import zipfile
import logging
from typing import Dict, Any, Tuple
from fastapi import APIRouter, Request, Body, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from sse_starlette.sse import EventSourceResponse
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

from app.core.config import settings, ProviderConfig
from app.core.state_manager import session_manager
from app.core.context import ServiceContext
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


router = APIRouter()

async def _initialize_session_context(params: GraphRunParams) -> Tuple[ServiceContext, Dict[str, ProviderConfig]]:
    """Centralizes LLM initialization and creates the service context."""
    init_params = LLMInitializationParams(
        debug_mode=params.debug_mode,
        coder_debug_mode=params.coder_debug_mode
    )
    (llm, llm_config), \
    (synth_llm, synth_config), \
    (summ_llm, summ_config), \
    (embed_model, embed_config) = await initialize_llms(init_params)

    services = ServiceContext(
        llm=llm,
        synthesizer_llm=synth_llm,
        summarizer_llm=summ_llm,
        embeddings_model=embed_model
    )
    
    configs = {
        "llm_config": llm_config,
        "synthesizer_llm_config": synth_config,
        "summarizer_llm_config": summ_config,
        "embeddings_config": embed_config,
    }
    return services, configs

@router.get("/config")
async def get_app_config():
    """Provides essential configuration details to the frontend."""
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
        service_context, llm_configs = await _initialize_session_context(payload.params)
        logging.info("--- [SETUP] Initialized LLMs and Service Context successfully.")
        
        request_is_code_chain = prompt_service.create_chain(service_context.llm, "request_is_code")
        
        is_code_str = (await request_is_code_chain.ainvoke({"request": user_prompt})).lower().strip()
        is_code = "true" in is_code_str

        if payload.params.coder_debug_mode: is_code = True
        elif payload.params.debug_mode: is_code = False

        logging.info(f"--- [SETUP] Request is code: {is_code}")

        logging.info("--- [SETUP] Attempting to build graph workflow...")
        graph, initial_state_components = await build_graph_workflow(
            payload.params, 
            service_context.llm, llm_configs['llm_config'], 
            service_context.synthesizer_llm, 
            service_context.summarizer_llm, 
            service_context.embeddings_model, 
            is_code
        )
        logging.info("--- [SETUP] Graph workflow built successfully.")
        
        initial_state = {
            "previous_solution": "", "chat_history": [], "original_request": user_prompt,
            "current_problem": user_prompt, "epoch": 0, "max_epochs": payload.params.num_epochs,
            "params": payload.params, "agent_outputs": {}, "memory": {}, "critiques": {},
            "final_solution": None, "perplexity_history": [], "all_rag_documents": [],
            "academic_papers": None, "is_code_request": is_code,
            "modules": [], "synthesis_context_queue": [], "synthesis_execution_success": True,
            **llm_configs,
            **initial_state_components
        }
        
        loggable_state = {k: v for k, v in initial_state.items() if not isinstance(v, GraphRunParams)}
        loggable_state['params'] = initial_state['params'].model_dump()
        logging.info(f"--- Graph Initial State ---\n{json.dumps(loggable_state, indent=2, default=str)}")
        
        session_id = session_manager.create_session(initial_state, service_context)
        
        logging.info(f"--- Starting Execution (Epochs: {payload.params.num_epochs}) ---")

        try:
            recursion_limit = (payload.params.num_epochs * len(graph.nodes)) + 50

            # PATTERN: DI - Pass session_id via 'configurable' to enable nodes to locate their ServiceContext.
            async for output in graph.astream(
                initial_state, 
                {'recursion_limit': recursion_limit, "configurable": {"session_id": session_id}}
            ):
                for node_name, node_output in output.items():
                    logging.info(f"--- Node Finished: {node_name} ---")
                    if isinstance(node_output, dict) and node_output:
                        await session_manager.update_session_state(session_id, node_output)
        
        except AgentExecutionError as e:
            error_message = f"Execution halted due to a critical failure in agent '{e.node_id}'. Reason: {e.message}"
            logging.error("--- FATAL EXECUTION ERROR ---", extra={"ui_extra": None})
            logging.error(error_message, extra={"ui_extra": None})
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

@router.post("/continue_run_from_state")
async def continue_run_from_state(payload: dict = Body(...)):
    """
    Loads a QNN state and continues the training process for a specified
    number of additional epochs. This rebuilds the full graph with reflection
    loops and hydrates it with the agent prompts from the imported state.
    """
    logging.info("--- [CONTINUE] Received request to continue run from imported state. ---")
    try:
        imported_state = payload.get("imported_state")
        if not imported_state:
            return JSONResponse(content={"error": "Payload must contain 'imported_state'."}, status_code=400)

        # Ensure params are parsed into the Pydantic model
        params_dict = imported_state.get("params", {})
        if isinstance(params_dict, dict):
            params = GraphRunParams(**params_dict)
        elif isinstance(params_dict, GraphRunParams):
            params = params_dict
        else:
             return JSONResponse(content={"error": "Invalid 'params' format in imported state."}, status_code=400)
        
        # We need to re-serialize the params back into the dict after potential model creation.
        imported_state["params"] = params
        
        # Extract key components from the loaded state
        existing_prompts = imported_state.get("all_layers_prompts")
        existing_personas = imported_state.get("agent_personas")
        existing_problems = imported_state.get("decomposed_problems")

        if not all([params, existing_prompts, existing_personas, existing_problems]):
            return JSONResponse(content={"error": "Imported state is missing necessary components for continuation."}, status_code=400)
        
        is_code = imported_state.get("is_code_request", False)

        service_context, llm_configs = await _initialize_session_context(params)
        
        imported_state.update(llm_configs)

        graph, _ = await build_graph_workflow(
            params, 
            service_context.llm, llm_configs['llm_config'], 
            service_context.synthesizer_llm, 
            service_context.summarizer_llm, 
            service_context.embeddings_model, 
            is_code,
            existing_prompts=existing_prompts,
            existing_personas=existing_personas,
            existing_problems=existing_problems
        )
        
        # The loaded state is our initial state for the continued run.
        session_id = session_manager.create_session(imported_state, service_context)
        
        logging.info(f"--- Starting Continued Execution (Epochs: {params.num_epochs}) ---")

        try:
            recursion_limit = (params.num_epochs * len(graph.nodes)) + 50
            async for output in graph.astream(
                imported_state, 
                {'recursion_limit': recursion_limit, "configurable": {"session_id": session_id}}
            ):
                for node_name, node_output in output.items():
                    logging.info(f"--- Node Finished: {node_name} ---")
                    if isinstance(node_output, dict) and node_output:
                        await session_manager.update_session_state(session_id, node_output)
        
        except AgentExecutionError as e:
            error_message = f"Execution halted due to a critical failure in agent '{e.node_id}'. Reason: {e.message}"
            logging.error("--- FATAL EXECUTION ERROR ---", extra={"ui_extra": None})
            logging.error(error_message, extra={"ui_extra": None})
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
        error_message = f"An unexpected error occurred during continuation run setup: {e}"
        tb_str = traceback.format_exc()
        logging.critical(f"--- FATAL ERROR DURING CONTINUATION SETUP ---", exc_info=True)
        return JSONResponse(content={"message": error_message, "traceback": tb_str}, status_code=500)


@router.post("/run_inference_from_state")
async def run_inference_from_state(payload: dict = Body(...)):
    """Runs inference on an imported QNN state without further training."""
    logging.info("--- [INFERENCE-ONLY] Received request to run inference from imported state. ---")
    session_id = None
    try:
        imported_state = payload.get("imported_state")
        user_prompt = payload.get("prompt")
        params_dict = imported_state.get("params", {})

        if not imported_state or not user_prompt:
            return JSONResponse(content={"error": "Invalid payload. 'imported_state' and 'prompt' are required."}, status_code=400)
        
        params = GraphRunParams(**params_dict)
        service_context, llm_configs = await _initialize_session_context(params)

        all_layers_prompts = imported_state.get("all_layers_prompts", [])
        if not all_layers_prompts:
            return JSONResponse(content={"error": "Imported state must contain 'all_layers_prompts'."}, status_code=400)

        inference_state = imported_state.copy()
        inference_state.update({
            "original_request": user_prompt, "current_problem": user_prompt,
            "agent_outputs": {}, "is_code_request": True, 
            "synthesis_context_queue": imported_state.get("synthesis_context_queue", []),
            **llm_configs
        })

        # ARCH: Create temporary session to satisfy node DI pattern during stateless inference.
        # This session will auto-expire via TTLCache policy.
        session_id = session_manager.create_session(inference_state, service_context)

        # PHASE: Graph Definition - Statically define the inference workflow topology.
        workflow = StateGraph(GraphState)

        # 1. Node Registration
        for i, layer in enumerate(all_layers_prompts):
            for j in range(len(layer)):
                node_id = f"agent_{i}_{j}"
                workflow.add_node(node_id, create_agent_node(node_id))

        workflow.add_node("synthesis", create_synthesis_node())

        # 2. Edge Connectivity
        layer_node_ids = [[f"agent_{i}_{j}" for j in range(len(all_layers_prompts[i]))] for i in range(len(all_layers_prompts))]
        
        # For parallel execution of the first layer
        workflow.set_entry_point(layer_node_ids[0])
        
        # Correct for for parallel execution
        for i in range(len(layer_node_ids) - 1):
            workflow.add_edge(layer_node_ids[i], layer_node_ids[i+1])
        
        workflow.add_edge(layer_node_ids[-1], "synthesis")
        workflow.add_edge("synthesis", END)

        graph = workflow.compile()
        logging.info("Inference graph compiled.", extra={'ui_extra': {'type': 'graph', 'data': graph.get_graph().draw_ascii()}})
        
        # PATTERN: DI - Pass temporary session_id to enable node service location.
        final_result_node = None
        async for output in graph.astream(inference_state, {"configurable": {"session_id": session_id}}):
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
    session_object = session_manager.get_session(session_id)
    if not session_object or not session_object.get('state'):
        return JSONResponse(content={"error": "Session state not found or empty."}, status_code=404)

    state_to_export = session_object['state'].copy()
    
    if 'params' in state_to_export and isinstance(state_to_export['params'], GraphRunParams):
        state_to_export['params'] = state_to_export['params'].model_dump()
    if 'all_rag_documents' in state_to_export:
        docs = state_to_export.get('all_rag_documents', [])
        state_to_export['all_rag_documents'] = [doc.dict() for doc in docs if isinstance(doc, Document)]

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
        
        if 'all_rag_documents' in imported_data:
            imported_data['all_rag_documents'] = [Document(**doc) for doc in imported_data['all_rag_documents']]
        
        if 'params' in imported_data:
            imported_data['params'] = GraphRunParams(**imported_data['params'])
        else:
             return JSONResponse(content={"error": "Imported QNN state must contain 'params' section."}, status_code=400)
        
        service_context, _ = await _initialize_session_context(imported_data['params'])

        session_id = session_manager.create_session(imported_data, service_context)
        logging.info(f"--- [IMPORT] Successfully imported QNN file. New Session ID: {session_id} ---")
        
        return JSONResponse(content={
            "message": "QNN file imported successfully.",
            "session_id": session_id,
            "imported_params": imported_data.get("params", {}).model_dump()
        })
    except Exception as e:
        error_message = f"Failed to import QNN file: {e}"
        logging.error(error_message, exc_info=True)
        return JSONResponse(content={"message": error_message, "traceback": traceback.format_exc()}, status_code=500)


@router.post("/chat")
async def chat_with_index(payload: dict = Body(...)):
    """
    Handles real-time, streaming chat with the RAG index of an active session.

    Expects a payload containing `session_id` and a `message`. It retrieves the
    session's ServiceContext, performs a RAG query, and streams the LLM's
    response back to the client. The chat history is atomically updated.
    """
    session_id = payload.get("session_id")
    session_object = session_manager.get_session(session_id)
    if not session_object:
        return JSONResponse(content={"error": "Invalid session ID"}, status_code=404)

    services: ServiceContext = session_object.get("services")
    raptor_index = services.raptor_index
    llm = services.llm
    
    if not raptor_index or not llm:
        return JSONResponse(content={"error": "RAG index or LLM not found in this session."}, status_code=500)

    async def stream_response():
        message = payload.get("message")
        
        # CRITICAL: Lock to ensure chat history is updated atomically.
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
            
            # CRITICAL: Lock again to append the final AI response.
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

    services: ServiceContext = session_object.get("services")
    raptor_index = services.raptor_index
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

    try:
        logging.info("--- [HARVEST] Initiating Final Harvest Process ---")
        state_for_graph_run = {}
        # Atomically update state with chat history and get a snapshot for the graph run.
        async with session_object["lock"]:
            state = session_object["state"]
            chat_history = state.get("chat_history", [])
            if chat_history:
                chat_docs = [
                    Document(page_content=f"User: {chat_history[i-1]['content']}\nAI: {turn['content']}",
                             metadata={"source": "chat", "turn": i // 2})
                    for i, turn in enumerate(chat_history) if turn['role'] == 'ai'
                ]
                if "all_rag_documents" not in state or not isinstance(state["all_rag_documents"], list):
                    state["all_rag_documents"] = []
                state["all_rag_documents"].extend(chat_docs)

            state_for_graph_run = state.copy()
            
        harvest_workflow = StateGraph(GraphState)
        
        # ARCH: The update_rag_node factory returns a specialized lambda for this post-run harvest graph.
        _, update_rag_node_final = create_update_rag_index_node()
        harvest_workflow.add_node("update_rag_index_final", update_rag_node_final)
        
        num_questions = int(state_for_graph_run["params"].num_questions_for_harvest)
        harvest_node = create_final_harvest_node(num_questions)
        harvest_workflow.add_node("final_harvest", harvest_node)

        harvest_workflow.set_entry_point("update_rag_index_final")
        harvest_workflow.add_edge("update_rag_index_final", "final_harvest")
        harvest_workflow.add_edge("final_harvest", END)
        
        harvest_graph = harvest_workflow.compile()
        
        logging.info("--- [HARVEST] Executing Harvest Graph... ---")
        async for output in harvest_graph.astream(
            state_for_graph_run, 
            {"configurable": {"session_id": session_id}}
        ):
            for node_name, node_output in output.items():
                logging.info(f"--- [HARVEST] Node Finished: {node_name} ---")
                if isinstance(node_output, dict) and node_output:
                    # Use the new thread-safe method for state updates
                    await session_manager.update_session_state(session_id, node_output)
        
        # After the run, get the final state to find the report
        updated_session_object = session_manager.get_session(session_id)
        if not updated_session_object:
             return JSONResponse(content={"message": "Session expired during harvest"}, status_code=500)

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
