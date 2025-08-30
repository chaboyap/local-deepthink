# app/graph/nodes/synthesis_nodes.py

import asyncio
import json
import logging
from app.graph.state import GraphState
from app.services.prompt_service import prompt_service
from app.core.config import settings
from app.services.structured_output_adapter import get_structured_output
from app.services.schemas import SynthesisOutput, GeneratedQuestions, CodeSynthesisOutput
from app.utils.exceptions import AgentExecutionError
from app.core.context import ServiceContext

async def _synthesis_node_logic(state: GraphState, services: ServiceContext) -> dict:
    logging.info("--- [FORWARD PASS] Synthesizing Final Solution ---")
    is_code = state.get("is_code_request", False)
    synthesizer_llm = services.synthesizer_llm
    synthesizer_llm_config = state["synthesizer_llm_config"]
    timeout = settings.hyperparameters.timeouts.synthesis_timeout_seconds
    
    last_layer_idx = len(state['all_layers_prompts']) - 1
    last_layer_outputs = [v for k, v in state["agent_outputs"].items() if k.startswith(f"agent_{last_layer_idx}_")]

    if not last_layer_outputs:
        logging.warning("SYNTHESIS_NODE_WARNING: No inputs from the final agent layer.")
        return {"final_solution": {"error": "Synthesis failed: No inputs from the final agent layer."}}

    if is_code:
        synthesis_context = "\n\n".join(state.get("synthesis_context_queue", []))
        if not synthesis_context:
            synthesis_context = "No modules have been successfully built yet."
        logging.info(f"LOG: Providing synthesis agent with context from {len(state.get('synthesis_context_queue', []))} modules.")
    else:
        synthesis_context = ""

    synthesis_input = { "original_request": state["original_request"], "agent_solutions": json.dumps(last_layer_outputs), "synthesis_context": synthesis_context }
    
    try:
        template_name = "code_synthesis" if is_code else "synthesis"
        prompt_template = prompt_service.get_template(template_name)
        full_prompt_for_logging = prompt_template.format(**synthesis_input)
        logging.info(f"--- SYNTHESIZER FULL INPUT ---\n{full_prompt_for_logging}")
    except Exception as e:
        logging.warning(f"Could not format synthesizer prompt for logging: {e}")

    try:
        if is_code:
            solution_obj = await asyncio.wait_for(
                get_structured_output(
                    llm=synthesizer_llm, provider_config=synthesizer_llm_config,
                    prompt_template=prompt_service.get_template("code_synthesis"),
                    input_data=synthesis_input, pydantic_schema=CodeSynthesisOutput
                ), timeout=timeout)
            solution = solution_obj.model_dump() if solution_obj else None
        else:
            solution_obj = await asyncio.wait_for(
                get_structured_output(
                    llm=synthesizer_llm, provider_config=synthesizer_llm_config,
                    prompt_template=prompt_service.get_template("synthesis"),
                    input_data={"original_request": state["original_request"], "agent_solutions": json.dumps(last_layer_outputs)},
                    pydantic_schema=SynthesisOutput
                ), timeout=timeout)
            solution = solution_obj.model_dump() if solution_obj else None

        if not solution: raise ValueError("LLM returned malformed or empty solution.")

        logging.info("SUCCESS: Synthesis complete, final solution generated.")
        logging.info(f"--- FINAL SYNTHESIZED SOLUTION ---\n{json.dumps(solution, indent=2)}")
        
        return {"final_solution": solution}

    except Exception as e:
        error_msg = f"Synthesis failed on all attempts. Halting graph execution. Error: {e}"
        logging.error(f"SYNTHESIS_NODE_ERROR: {error_msg}")
        raise AgentExecutionError(message="Synthesis node failed to produce a valid output.", node_id="synthesis")
        
def create_synthesis_node():
    """Creates the node that synthesizes the final layer of agent outputs into a single solution."""
    async def synthesis_node_wrapper(state: GraphState) -> dict:
        services: ServiceContext = state.get("services") # type: ignore
        if not services:
            raise RuntimeError(f"ServiceContext not found for synthesis node")
        
        return await _synthesis_node_logic(state, services)
            
    return synthesis_node_wrapper

async def _final_harvest_node_logic(state: GraphState, services: ServiceContext, num_questions: int) -> dict:
    logging.info("--- [FINAL HARVEST] Starting Knowledge Harvest Process ---")
    raptor_index = services.raptor_index
    if not raptor_index:
        logging.error("HARVEST_NODE_ERROR: No RAG index found. Cannot proceed.")
        return {}

    timeout = settings.hyperparameters.timeouts.synthesis_timeout_seconds
    
    logging.info(f"Interrogating RAG Index to generate {num_questions} expert questions...")
    user_questions = [doc["content"] for doc in state.get("chat_history", []) if doc["role"] == "user"]
    
    try:
        questions_obj = await asyncio.wait_for(
            get_structured_output(
                llm=services.llm, provider_config=state["llm_config"],
                prompt_template=prompt_service.get_template("interrogator"),
                input_data={
                    "original_request": state["original_request"],
                    "further_questions": "\n".join(user_questions),
                    "num_questions": num_questions
                },
                pydantic_schema=GeneratedQuestions
            ), timeout=timeout
        )
        questions = questions_obj.questions if questions_obj else []
        if not questions: raise ValueError("Interrogator LLM returned no questions.")
        logging.info(f"--- [HARVEST] Generated {len(questions)} expert questions for the final report. ---")
    except Exception as e:
        logging.error(f"HARVEST_NODE_ERROR: Failed to generate harvest questions: {e}. Aborting.")
        return {}
        
    logging.info(f"--- [HARVEST] Generating {len(questions)} academic-style papers... ---")
    paper_formatter_chain = prompt_service.create_chain(services.synthesizer_llm, "paper_formatter")
    
    async def generate_paper(question: str) -> tuple[str, str] | None:
        try:
            retrieved_docs = await asyncio.to_thread(raptor_index.retrieve, question, k=40)
            rag_context = "\n\n---\n\n".join(d.page_content for d in retrieved_docs)[:400000] 
            
            paper_content = await asyncio.wait_for(
                paper_formatter_chain.ainvoke({"question": question, "rag_context": rag_context}),
                timeout=timeout * 1.5
            )
            logging.info(f"SUCCESS: Generated paper for question: '{question[:60]}...'")
            return question, paper_content
        except Exception as e:
            logging.error(f"HARVEST_PAPER_ERROR: Failed for question '{question[:60]}...'. Error: {e}")
            return None

    paper_results = await asyncio.gather(*(generate_paper(q) for q in questions))
    papers = dict(filter(None, paper_results))
    
    logging.info(f"--- [FINAL HARVEST] Finished. Generated {len(papers)} of {len(questions)} papers. ---")
    return {"academic_papers": papers}
    
def create_final_harvest_node(num_questions):
    """Creates the final harvest node for generating the comprehensive report."""
    async def final_harvest_node_wrapper(state: GraphState) -> dict:
        services: ServiceContext = state.get("services") # type: ignore
        if not services:
            raise RuntimeError(f"ServiceContext not found for final harvest node")
        
        return await _final_harvest_node_logic(state, services, num_questions)

    return final_harvest_node_wrapper