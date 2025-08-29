# app/graph/nodes/execution_nodes.py

import io
import re
import logging
from contextlib import redirect_stdout, redirect_stderr
from langchain_core.runnables import RunnableConfig
from app.graph.state import GraphState
from app.services.prompt_service import prompt_service
from app.core.state_manager import session_manager
from app.core.context import ServiceContext

def execute_code_in_sandbox(code: str) -> (bool, str):
    """
    Executes a string of Python code and captures its stdout/stderr.
    Returns a tuple of (success: bool, output: str).
    """
    if not code:
        return True, "No code to execute."
        
    # Extract code from markdown block if present
    code_match = re.search(r"```(?:python\n)?([\s\S]*?)```", code)
    if code_match:
        code = code_match.group(1).strip()

    output_buffer = io.StringIO()
    try:
        with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
            # Using a restricted globals dict for a little more safety
            exec(code, {'__builtins__': {
                'print': print, 'range': range, 'len': len, 'str': str, 'int': int, 'float': float, 
                'list': list, 'dict': dict, 'set': set, 'tuple': tuple, 'True': True, 'False': False, 'None': None
            }})
        return True, output_buffer.getvalue()
    except Exception as e:
        return False, f"{output_buffer.getvalue()}\n\nERROR: {type(e).__name__}: {e}"

async def _code_execution_node_logic(state: GraphState, services: ServiceContext):
    if not state.get("is_code_request"):
        return {"synthesis_execution_success": True} 

    logging.info("--- [SANDBOX] Testing Synthesized Code ---")
    synthesized_code = state.get("final_solution", {}).get("proposed_solution", "")
    
    success, output = execute_code_in_sandbox(synthesized_code)
    
    logging.info(f"--- [SANDBOX] Synthesized Code Result: {'Success' if success else 'Failure'} ---")
    logging.info(output)

    if success:
        module_card_chain = prompt_service.create_chain(services.llm, "module_card")
        module_card = await module_card_chain.ainvoke({"code": synthesized_code})
        
        logging.info("--- [MODULE CARD] ---")
        logging.info(module_card)
        
        new_modules = state.get("modules", []) + [{"code": synthesized_code, "card": module_card}]
        new_context_queue = state.get("synthesis_context_queue", []) + [module_card]
        
        return {
            "synthesis_execution_success": True,
            "modules": new_modules,
            "synthesis_context_queue": new_context_queue
        }
    else:
        return {"synthesis_execution_success": False}

def create_code_execution_node():
    """Creates node to execute and validate the synthesized code."""
    async def code_execution_node_wrapper(state: GraphState, config: RunnableConfig):
        session_id = config["configurable"]["session_id"]
        session = session_manager.get_session(session_id)
        if not session:
            raise RuntimeError(f"Session {session_id} not found for code execution node")
        
        services: ServiceContext = session["services"]
        return await _code_execution_node_logic(state, services)
                   
    return code_execution_node_wrapper