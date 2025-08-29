# app/api/endpoints.py

import asyncio
import io
import json
import re
import traceback
import zipfile
import logging

from fastapi import APIRouter, Request, Body, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from sse_starlette.sse import EventSourceResponse
from langchain_core.documents import Document

from app.core.config import settings
from app.core.state_manager import session_manager
from app.core.context import ServiceContext
from app.services.prompt_service import prompt_service
from app.api.schemas import GraphRunPayload, GraphRunParams
from app.utils.exceptions import AgentExecutionError
from app.services.graph_orchestration_service import graph_orchestration_service

router = APIRouter()

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
            "mbti_archetypes": hyperparams.default_mbti_selection,
            }
        return JSONResponse(content={
            "default_model": settings.llm_providers.main_llm.model_name,
            "defaults": frontend_defaults,
            })
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not load configuration.")


@router.post("/build_and_run_graph")
async def build_and_run_graph(payload: GraphRunPayload):
    """Delegates graph creation and execution to the orchestration service."""
    try:
        logging.info(f"--- API received request to start graph run ---")
        result = await graph_orchestration_service.create_and_run_new_graph(payload.params)
        return JSONResponse(content=result)
    except AgentExecutionError as e:
        error_message = f"Execution halted in node '{e.node_id}'. Reason: {e.message}"
        logging.error(f"--- AGENT EXECUTION ERROR --- \n{error_message}")
        raise HTTPException(status_code=500, detail=error_message)
    except Exception as e:
        error_message = "An unexpected error occurred during graph execution."
        logging.critical(f"--- FATAL ERROR IN GRAPH EXECUTION ---", exc_info=True)
        raise HTTPException(status_code=500, detail={"message": error_message, "traceback": traceback.format_exc()})

@router.post("/continue_run_from_state")
async def continue_run_from_state(payload: dict = Body(...)):
    """Delegates continuation logic to the orchestration service."""
    try:
        imported_state = payload.get("imported_state")
        if not imported_state:
            raise HTTPException(status_code=400, detail="Payload must contain 'imported_state'.")
        logging.info("--- API received request to continue run from state. ---")
        result = await graph_orchestration_service.continue_graph_run(imported_state)
        return JSONResponse(content=result)
    except (TypeError, Exception) as e:
        error_message = "An unexpected error occurred during continuation."
        logging.critical(f"--- FATAL ERROR DURING CONTINUATION ---", exc_info=True)
        raise HTTPException(status_code=500, detail={"message": error_message, "traceback": traceback.format_exc()})


@router.post("/run_inference_from_state")
async def run_inference_from_state(payload: dict = Body(...)):
    """Delegates inference logic to the orchestration service."""
    try:
        imported_state = payload.get("imported_state")
        user_prompt = payload.get("prompt")
        if not all([imported_state, user_prompt]):
            raise HTTPException(status_code=400, detail="Payload must contain 'imported_state' and 'prompt'.")

        logging.info("--- API received request for inference-only run. ---")
        result = await graph_orchestration_service.run_inference_only(imported_state, user_prompt)
        return JSONResponse(content=result)
    except Exception as e:
        error_message = "An unexpected error occurred during inference."
        logging.error(error_message, exc_info=True)
        raise HTTPException(status_code=500, detail={"message": error_message, "traceback": traceback.format_exc()})

@router.post("/harvest")
async def harvest_session(payload: dict = Body(...)):
    """Delegates harvest logic to the orchestration service."""
    try:
        session_id = payload.get("session_id")
        if not session_id:
            raise HTTPException(status_code=400, detail="Payload must contain 'session_id'.")
            
        logging.info(f"--- API received request to harvest session {session_id}. ---")
        result = await graph_orchestration_service.harvest_and_generate_report(session_id)
        return JSONResponse(content=result)
    except Exception as e:
        error_message = "An error occurred during harvest."
        logging.error(f"--- FATAL ERROR DURING HARVEST ---", exc_info=True)
        raise HTTPException(status_code=500, detail={"message": error_message, "traceback": traceback.format_exc()})


@router.get("/export_qnn/{session_id}")
async def export_qnn(session_id: str):
    """Exports the current state of a session graph to a JSON file."""
    session_object = session_manager.get_session(session_id)
    if not session_object or not session_object.get('state'):
        raise HTTPException(status_code=404, detail="Session state not found or empty.")
    state_to_export = session_object['state'].copy()
    if 'params' in state_to_export and isinstance(state_to_export['params'], GraphRunParams):
        state_to_export['params'] = state_to_export['params'].model_dump()
    if 'all_rag_documents' in state_to_export:
        state_to_export['all_rag_documents'] = [doc.dict() for doc in state_to_export.get('all_rag_documents', []) if isinstance(doc, Document)]
    logging.info(f"--- [EXPORT] Exporting QNN for session {session_id} ---")
    return JSONResponse(content=state_to_export, headers={"Content-Disposition": f"attachment; filename=qnn_state_{session_id}.json"})

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
             raise HTTPException(status_code=400, detail="Imported QNN state must contain 'params' section.")
             
        service_context, _ = await graph_orchestration_service._initialize_context(imported_data['params']) # Re-use the helper
        session_id = session_manager.create_session(imported_data, service_context)
        logging.info(f"--- [IMPORT] Successfully imported QNN file. New Session ID: {session_id} ---")
        return JSONResponse(content={"message": "QNN file imported successfully.", "session_id": session_id, "imported_params": imported_data.get("params", {}).model_dump()})
    except Exception as e:
        raise HTTPException(status_code=500, detail={"message": f"Failed to import QNN file: {e}", "traceback": traceback.format_exc()})


@router.post("/chat")
async def chat_with_index(payload: dict = Body(...)):
    """
    Handles real-time, streaming chat with the RAG index of an active session.

    Expects a payload containing `session_id` and a `message`. It retrieves the
    session's ServiceContext, performs a RAG query, and streams the LLM's
    response back to the client. The chat history is atomically updated.
    """
    session_id = payload.get("session_id")
    message = payload.get("message")
    session_object = session_manager.get_session(session_id)
    if not session_object: raise HTTPException(status_code=404, detail="Invalid session ID")
    services: ServiceContext = session_object.get("services")
    if not services or not services.raptor_index or not services.llm:
        raise HTTPException(status_code=500, detail="RAG index or LLM not found in this session.")

    async def stream_response():
        # CRITICAL: Lock to ensure chat history is updated atomically.
        async with session_object["lock"]:
            session_object["state"]["chat_history"].append({"role": "user", "content": message})
        try:
            retrieved_docs = await asyncio.to_thread(services.raptor_index.retrieve, message, k=10)
            context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
            chat_chain = prompt_service.create_chain(services.llm, "rag_chat", is_chat_prompt=True)
            full_response = ""
            async for chunk in chat_chain.astream({"context": context, "question": message}):
                content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                yield content
                full_response += content
            # CRITICAL: Lock again to append the final AI response.
            async with session_object["lock"]:
                if session_object["state"]["chat_history"] and session_object["state"]["chat_history"][-1]['role'] == 'user':
                    session_object["state"]["chat_history"].append({"role": "ai", "content": full_response})
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