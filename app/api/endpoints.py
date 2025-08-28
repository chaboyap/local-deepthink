# app/api/endpoints.py

import asyncio
import io
import json
import re
import traceback
import zipfile
import logging
from typing import Dict, Any
from fastapi import APIRouter, Request, Body
from fastapi.responses import JSONResponse, StreamingResponse
from sse_starlette.sse import EventSourceResponse
from langchain_core.documents import Document

from app.core.config import settings
from app.core.state_manager import session_manager
from app.graph.nodes import create_final_harvest_node, create_update_rag_index_node
from app.graph.workflow import build_graph_workflow
from app.services.llm_service import initialize_llms
from app.services.prompt_service import prompt_service
from app.api.schemas import GraphRunPayload
from app.utils.exceptions import AgentExecutionError

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
            "critique_strategy": hyperparams.critique_strategy,
            "learning_rate": hyperparams.learning_rate,
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
        (llm, llm_config), \
        (synthesizer_llm, synthesizer_llm_config), \
        (summarizer_llm, summarizer_llm_config), \
        (embeddings_model, embeddings_config) = await initialize_llms(payload.params)
        
        logging.info("--- [SETUP] Initialized LLMs successfully.")
        
        request_is_code_chain = prompt_service.create_chain(llm, "request_is_code")
        
        is_code_str = (await request_is_code_chain.ainvoke({"request": user_prompt})).lower()
        if "true" in is_code_str:
            is_code = True
        elif "false" in is_code_str:
            is_code = False
        else:
            #Maybe we retry the check? Gotta setup the framework for this however.
            raise RuntimeError(f"CRITICAL: is_code_str is not true or false: HALTING!")
        
        logging.info(f"--- [SETUP] Request is code: {is_code}")

        logging.info("--- [SETUP] Attempting to build graph workflow...")
        graph, initial_state_components = await build_graph_workflow(
            payload.params, llm, llm_config, synthesizer_llm, summarizer_llm, embeddings_model
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
            "significant_progress_made": False,
            "raptor_index": None,
            "all_rag_documents": [],
            "academic_papers": None,
            "critique_prompt": prompt_service.get_template("initial_global_critique"),
            "individual_critique_prompt": prompt_service.get_template("initial_individual_critique"),
            "assessor_prompt": prompt_service.get_template("initial_assessor"),
            "is_code_request": is_code,
            "llm": llm,
            "llm_config": llm_config,
            "summarizer_llm": summarizer_llm,
            "summarizer_llm_config": summarizer_llm_config,
            "embeddings_model": embeddings_model,
            "embeddings_config": embeddings_config,
            "synthesizer_llm": synthesizer_llm,
            "synthesizer_llm_config": synthesizer_llm_config,
            **initial_state_components
        }
        
        loggable_state = {k: (f"<{v.__class__.__name__}>" if hasattr(v, '__class__') and 'langchain' in str(v.__class__) else v) for k, v in initial_state.items()}
        logging.info(f"--- Graph Initial State ---\n{json.dumps(loggable_state, indent=2, default=str)}")
        
        session_id = session_manager.create_session(initial_state)
        
        logging.info(f"--- Starting Execution (Epochs: {payload.params.num_epochs}) ---")

        try:
            recursion_limit = (payload.params.num_epochs * len(graph.nodes)) + 50
            # Get the initial state from the session object to pass to astream
            initial_session_object = session_manager.get_session(session_id)
            if not initial_session_object:
                raise RuntimeError("Failed to create and retrieve session immediately.")
                
            # The graph updates the session state in-place via the session_manager
            async for output in graph.astream(initial_session_object['state'], {'recursion_limit': recursion_limit}):
                for node_name, node_output in output.items():
                    logging.info(f"--- Node Finished: {node_name} ---")
                    
                    if isinstance(node_output, dict) and node_output:
                        session_manager.update_session_state(session_id, node_output)
        
        except AgentExecutionError as e:
            error_message = f"Execution halted due to a critical failure in agent '{e.node_id}'. Reason: {e.message}"
            logging.error("--- FATAL EXECUTION ERROR ---")
            logging.error(error_message)
            return JSONResponse(
                content={"message": error_message}, 
                status_code=500
            )
        
        # Retrieve the final state after the graph run completes
        final_session_object = session_manager.get_session(session_id)
        if final_session_object:
             final_state_value = final_session_object['state']
             # set_final_state is a good practice if you plan to do more with the session,
             # but is optional here as the state is already updated.
             session_manager.set_final_state(session_id, final_state_value)
        else:
            logging.error(f"Session {session_id} expired or was lost before completion.")
            return JSONResponse(content={"message": f"Session {session_id} could not be found after execution."}, status_code=404)

        if is_code:
            final_solution = final_state_value.get("final_solution", {})
            proposed_solution = final_solution.get("proposed_solution", "")
            reasoning = final_solution.get("reasoning", "")

            # Add a check for an empty solution, which can happen.
            if proposed_solution and proposed_solution.strip():
                return JSONResponse(content={
                    "message": "Code generation complete.",
                    "code_solution": proposed_solution,
                    "reasoning": reasoning
                })
            else:
                logging.warning("The 'proposed_solution' from the final graph state was empty. Returning a placeholder.")
                return JSONResponse(content={
                    "message": "Code generation complete.",
                    "code_solution": "# The agent network failed to produce a valid code solution.",
                    "reasoning": "The final synthesis step resulted in an empty or invalid output. This can occur when agents in the final layers fail to generate solutions, often due to overly complex sub-problems or model limitations.",
                    "warning": "The 'proposed_solution' from the final graph state was empty. Returning a placeholder."
                })
        else:
            logging.info("--- Agent Execution Finished. Pausing for User Chat. ---")
            return JSONResponse(content={"message": "Chat is now active.", "session_id": session_id})

    except Exception as e:
        error_message = f"An unexpected error occurred during graph setup: {e}"
        tb_str = traceback.format_exc()
        
        logging.critical(f"--- FATAL ERROR DURING GRAPH SETUP ---", exc_info=True)
        logging.critical(error_message)
        logging.critical(tb_str)
        
        return JSONResponse(
            content={"message": error_message, "traceback": tb_str}, 
            status_code=500
        )

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
        try:
            retrieved_docs = await asyncio.to_thread(raptor_index.retrieve, message, k=10)
            context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
            chat_chain = prompt_service.create_chain(llm, "rag_chat", is_chat_prompt=True)
            
            full_response = ""
            async for chunk in chat_chain.astream({"context": context, "question": message}):
                content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                yield content
                full_response += content
            
            # Acquire a lock BEFORE modifying the shared chat_history
            async with session_object["lock"]:
                current_state = session_object["state"]
                current_state["chat_history"].extend([
                    {"role": "user", "content": message},
                    {"role": "ai", "content": full_response}
                ])
                
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

                logging.info(f"--- [RAG PASS] Building Final RAPTOR Index with {len(chat_docs)} chat documents ---")
                update_rag_node = create_update_rag_index_node(state["summarizer_llm"], state["embeddings_model"])
                
                rag_update_output = await update_rag_node(state, end_of_run=True)
                session_manager.update_session_state(session_id, rag_update_output)

            num_questions = int(state["params"].num_questions_for_harvest)
            harvest_node = create_final_harvest_node(state["llm"], state["summarizer_llm"], num_questions)
            
            harvest_output = await harvest_node(state)
            session_manager.update_session_state(session_id, harvest_output)
            
            # Re-fetch state to ensure we have the latest updates
            updated_session_object = session_manager.get_session(session_id)
            academic_papers = updated_session_object["state"].get("academic_papers", {})

            if academic_papers:
                session_manager.store_report(session_id, academic_papers)
                logging.info(f"SUCCESS: Final report with {len(academic_papers)} papers created for session {session_id}.")
            
            return JSONResponse(content={"message": "Harvest complete."})
        
        except Exception as e:
            error_message = f"An error occurred during harvest: {e}"
            logging.error(f"--- FATAL ERROR DURING HARVEST ---", exc_info=True)
            logging.error(f"{error_message}\n{traceback.format_exc()}")
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