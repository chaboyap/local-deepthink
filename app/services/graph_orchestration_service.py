# app/services/graph_orchestration_service.py

import asyncio
import json
import logging
import traceback
from typing import Dict, Any, Tuple

from langgraph.graph import StateGraph, END
from langchain_core.documents import Document

from app.api.schemas import GraphRunParams
from app.core.config import settings, ProviderConfig
from app.core.context import ServiceContext
from app.core.state_manager import session_manager
from app.graph.nodes import (
    create_agent_node, create_synthesis_node,
    create_update_rag_index_node, create_final_harvest_node
)
from app.graph.state import GraphState
from app.graph.workflow import build_graph_workflow
from app.services.llm_service import initialize_llms, LLMInitializationParams
from app.services.prompt_service import prompt_service
from app.utils.exceptions import AgentExecutionError


class GraphOrchestrationService:
    """
    SRP: This service's single responsibility is to orchestrate the setup,
    building, execution, and harvesting of LangGraph agent graphs.
    It encapsulates all business logic, leaving the API endpoints as thin controllers.
    """

    async def create_and_run_new_graph(self, params: GraphRunParams) -> Dict[str, Any]:
        """Orchestrates the creation and execution of a new graph run."""
        service_context, llm_configs = await self._initialize_context(params)
        is_code = "true" in (await prompt_service.create_chain(service_context.llm, "request_is_code")
                            .ainvoke({"request": params.prompt})).lower().strip()

        if params.coder_debug_mode: is_code = True
        elif params.debug_mode: is_code = False
        
        logging.info("--- [SERVICE] Building graph workflow for new run...")
        graph, initial_state_components = await build_graph_workflow(
            params, service_context.llm, llm_configs['llm_config'],
            service_context.synthesizer_llm, service_context.summarizer_llm,
            service_context.embeddings_model, is_code
        )
        logging.info("--- [SERVICE] Graph workflow built successfully.")

        initial_state = self._build_initial_state(params.prompt, params, llm_configs, initial_state_components, is_code)
        session_id = session_manager.create_session(initial_state, service_context)
        
        final_state = await self._execute_graph_stream(session_id, graph, initial_state)
        return self._format_final_response(final_state, session_id)

    async def continue_graph_run(self, imported_state: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrates the continuation of a graph run from a previous state."""
        params = self._get_params_from_state(imported_state)
        service_context, llm_configs = await self._initialize_context(params)
        is_code = imported_state.get("is_code_request", False)
        
        imported_state.update(llm_configs)

        graph, _ = await build_graph_workflow(
            params, service_context.llm, llm_configs['llm_config'],
            service_context.synthesizer_llm, service_context.summarizer_llm,
            service_context.embeddings_model, is_code,
            existing_prompts=imported_state.get("all_layers_prompts"),
            existing_personas=imported_state.get("agent_personas"),
            existing_problems=imported_state.get("decomposed_problems")
        )
        
        session_id = session_manager.create_session(imported_state, service_context)
        
        final_state = await self._execute_graph_stream(session_id, graph, imported_state)
        return self._format_final_response(final_state, session_id)

    async def run_inference_only(self, imported_state: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        """Orchestrates an inference-only run, building a custom inference graph."""
        params = self._get_params_from_state(imported_state)
        service_context, llm_configs = await self._initialize_context(params)

        inference_state = imported_state.copy()
        inference_state.update({
            "original_request": prompt, "current_problem": prompt,
            "agent_outputs": {}, "is_code_request": True, 
            "synthesis_context_queue": imported_state.get("synthesis_context_queue", []),
            **llm_configs
        })
        
        graph = self._build_and_compile_inference_graph(inference_state["all_layers_prompts"])
        
        # ARCH: Create temporary session to satisfy node DI pattern during stateless inference.
        # This session will auto-expire via TTLCache policy.
        session_id = session_manager.create_session(inference_state, service_context)

        final_result_node = None
        # PATTERN: DI - Pass temporary session_id to enable node service location.
        async for output in graph.astream(inference_state, {"configurable": {"session_id": session_id}}):
             if "synthesis" in output:
                final_result_node = output["synthesis"]

        if not final_result_node:
            raise AgentExecutionError("Inference run failed before a solution could be synthesized.", "synthesis")

        synthesis_output = final_result_node.get("final_solution", {})
        return {
            "message": "Inference complete.",
            "code_solution": synthesis_output.get("proposed_solution", "No solution generated."),
            "reasoning": synthesis_output.get("reasoning", "No reasoning provided."),
            "is_inference": True
        }

    async def harvest_and_generate_report(self, session_id: str):
        """Orchestrates the final knowledge harvest process."""
        session_object = session_manager.get_session(session_id)
        if not session_object:
            raise ValueError(f"Session {session_id} not found for harvesting.")

        async with session_object["lock"]:
            state = session_object["state"]
            chat_history = state.get("chat_history", [])
            if chat_history:
                 chat_docs = [ Document(page_content=f"User: {chat_history[i-1]['content']}\nAI: {turn['content']}", metadata={"source": "chat", "turn": i // 2}) for i, turn in enumerate(chat_history) if turn['role'] == 'ai' ]
                 if "all_rag_documents" not in state: state["all_rag_documents"] = []
                 state["all_rag_documents"].extend(chat_docs)
            state_for_graph_run = state.copy()
        
        graph = self._build_and_compile_harvest_graph(state_for_graph_run["params"])
        
        final_state = await self._execute_graph_stream(session_id, graph, state_for_graph_run, is_harvest=True)
        
        academic_papers = final_state.get("academic_papers", {})
        if academic_papers:
            session_manager.store_report(session_id, academic_papers)
            logging.info(f"SUCCESS: Final report stored for session {session_id}.")
        return {"message": "Harvest complete."}


    # --- Private Helper Methods (DRY & SRP applied) ---

    async def _initialize_context(self, params: GraphRunParams) -> Tuple[ServiceContext, Dict[str, ProviderConfig]]:
        init_params = LLMInitializationParams(debug_mode=params.debug_mode, coder_debug_mode=params.coder_debug_mode)
        (llm, llm_config), (synth_llm, synth_config), (summ_llm, summ_config), (embed_model, embed_config) = await initialize_llms(init_params)
        services = ServiceContext(llm=llm, synthesizer_llm=synth_llm, summarizer_llm=summ_llm, embeddings_model=embed_model)
        configs = {"llm_config": llm_config, "synthesizer_llm_config": synth_config, "summarizer_llm_config": summ_config, "embeddings_config": embed_config}
        return services, configs

    def _build_initial_state(self, prompt: str, params: GraphRunParams, llm_configs: Dict, components: Dict, is_code: bool) -> Dict:
        initial_state = { "original_request": prompt, "current_problem": prompt, "params": params, "epoch": 0, "max_epochs": params.num_epochs, "previous_solution": "", "chat_history": [], "agent_outputs": {}, "memory": {}, "critiques": {}, "final_solution": None, "perplexity_history": [], "all_rag_documents": [], "academic_papers": None, "is_code_request": is_code, "modules": [], "synthesis_context_queue": [], "synthesis_execution_success": True, **llm_configs, **components }
        loggable_state = {k: v for k, v in initial_state.items() if not isinstance(v, GraphRunParams)}
        loggable_state['params'] = initial_state['params'].model_dump()
        logging.info(f"--- [SERVICE] Graph Initial State Built ---\n{json.dumps(loggable_state, indent=2, default=str)}")
        return initial_state

    async def _execute_graph_stream(self, session_id: str, graph: StateGraph, initial_state: Dict, is_harvest: bool = False) -> Dict:
        config = {"recursion_limit": 250, "configurable": {"session_id": session_id}}
        if not is_harvest:
             config['recursion_limit'] = (initial_state['max_epochs'] * len(graph.nodes)) + 50
        
        # PATTERN: DI - Pass session_id via 'configurable' to enable nodes to locate their ServiceContext.
        async for output in graph.astream(initial_state, config):
            for node_name, node_output in output.items():
                logging.info(f"--- Node Finished: {node_name} ---")
                if isinstance(node_output, dict) and node_output:
                    await session_manager.update_session_state(session_id, node_output)
        
        final_session_object = session_manager.get_session(session_id)
        if not final_session_object:
            raise ConnectionError(f"Session {session_id} expired or was lost during execution.")
        
        final_state = final_session_object['state']
        session_manager.set_final_state(session_id, final_state)
        return final_state

    def _format_final_response(self, final_state: Dict, session_id: str) -> Dict[str, Any]:
        if final_state.get("is_code_request"):
            solution = final_state.get("final_solution", {})
            return {"message": "Code generation complete.", "code_solution": solution.get("proposed_solution", "# No code generated."), "reasoning": solution.get("reasoning", "No reasoning provided."), "modules": final_state.get("modules", []), "session_id": session_id}
        else:
            return {"message": "Chat is now active.", "session_id": session_id}
            
    def _build_and_compile_inference_graph(self, all_layers_prompts: list) -> StateGraph:
        # PHASE: Graph Definition - Statically define the inference workflow topology.
        workflow = StateGraph(GraphState)
        
        # 1. Node Registration
        for i, layer in enumerate(all_layers_prompts):
            for j in range(len(layer)):
                workflow.add_node(f"agent_{i}_{j}", create_agent_node(f"agent_{i}_{j}"))
        workflow.add_node("synthesis", create_synthesis_node())

        # 2. Edge Connectivity
        layer_node_ids = [[f"agent_{i}_{j}" for j in range(len(p))] for i, p in enumerate(all_layers_prompts)]
        
        # Parallel execution of all nodes in a layer
        workflow.set_entry_point(layer_node_ids[0][0] if layer_node_ids[0] else "synthesis") # Fallback if no agents
        if layer_node_ids and len(layer_node_ids[0]) > 1:
            for node in layer_node_ids[0][1:]:
                workflow.add_edge(layer_node_ids[0][0], node)

        for i in range(len(layer_node_ids) - 1):
            for source_node in layer_node_ids[i]:
                 for dest_node in layer_node_ids[i+1]:
                      workflow.add_edge(source_node, dest_node)

        if layer_node_ids:
            for final_layer_node in layer_node_ids[-1]:
                workflow.add_edge(final_layer_node, "synthesis")
        
        workflow.add_edge("synthesis", END)
        graph = workflow.compile()
        logging.info("Inference graph compiled.", extra={'ui_extra': {'type': 'graph', 'data': graph.get_graph().draw_ascii()}})
        return graph

    def _build_and_compile_harvest_graph(self, params: GraphRunParams) -> StateGraph:
        harvest_workflow = StateGraph(GraphState)
        
        # ARCH: The update_rag_node factory returns a specialized lambda for this post-run harvest graph.
        _, update_rag_node_final = create_update_rag_index_node()
        harvest_workflow.add_node("update_rag_index_final", update_rag_node_final)
        num_questions = int(params.num_questions_for_harvest)
        harvest_node = create_final_harvest_node(num_questions)
        harvest_workflow.add_node("final_harvest", harvest_node)
        harvest_workflow.set_entry_point("update_rag_index_final")
        harvest_workflow.add_edge("update_rag_index_final", "final_harvest")
        harvest_workflow.add_edge("final_harvest", END)
        return harvest_workflow.compile()
    
    def _get_params_from_state(self, state: dict) -> GraphRunParams:
        params_dict = state.get("params", {})
        if isinstance(params_dict, dict):
            return GraphRunParams(**params_dict)
        elif isinstance(params_dict, GraphRunParams):
            return params_dict
        raise TypeError("Invalid 'params' format in imported state.")

# Create a singleton instance for the API to use
graph_orchestration_service = GraphOrchestrationService()