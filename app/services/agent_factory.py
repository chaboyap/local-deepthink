# app/services/agent_factory.py

import random
import logging
import names
from typing import List, Dict, Tuple
from app.services.prompt_service import prompt_service
from app.services.structured_output_adapter import get_structured_output
from app.services.schemas import AgentAnalysisOutput
from app.api.schemas import GraphRunParams
from app.core.config import ProviderConfig

class AgentFactory:
    """Handles the creation and initialization of agent prompts for all layers."""
    def __init__(self, llm, llm_config: ProviderConfig):
        self.llm = llm
        self.llm_config = llm_config
        self.agent_personas = {}

    async def create_all_agent_prompts(self, decomposed_problems_map: Dict, params: GraphRunParams) -> Tuple[List[List[str]], Dict]:
        """Generates the initial system prompts for all agents in the network."""
        word_vector_size = params.vector_word_size
        mbti_archetypes = params.mbti_archetypes
        user_prompt = params.prompt
        
        seeds = await self._generate_seed_verbs(user_prompt, mbti_archetypes, word_vector_size)
        logging.info(f"Seed verbs generated: {seeds}")

        all_layers_prompts = []
        
        # --- Layer 0 ---
        logging.info("--- Creating Layer 0 Agents ---")
        input_spanner_chain = prompt_service.create_chain(
            self.llm,
            "input_spanner",
            prompt_alignment=params.prompt_alignment,
            density=params.density
        )
        layer_0_prompts = []
        for j, (mbti, guiding_words) in enumerate(seeds.items()):
            agent_id = f"agent_0_{j}"
            self.agent_personas[agent_id] = {"mbti_type": mbti, "name": names.get_full_name()}
            sub_problem = decomposed_problems_map.get(agent_id, user_prompt)
            prompt = await input_spanner_chain.ainvoke({
                "mbti_type": mbti,
                "guiding_words": guiding_words,
                "sub_problem": sub_problem,
                "critique": "",
                "name": self.agent_personas[agent_id]["name"]
            })
            layer_0_prompts.append(prompt)
        all_layers_prompts.append(layer_0_prompts)


        # --- Subsequent Layers ---
        dense_spanner_chain = prompt_service.create_chain(
            self.llm,
            "dense_spanner",
            prompt_alignment=params.prompt_alignment,
            density=params.density,
        )
        
        total_agents_to_create = len(mbti_archetypes) * params.cot_trace_depth
        agent_name_list = [names.get_full_name() for _ in range(total_agents_to_create)]

        for i in range(1, params.cot_trace_depth):
            logging.info(f"--- Creating Layer {i} Agents ---")
            current_layer_prompts = []
            for j, agent_prompt in enumerate(all_layers_prompts[i-1]):
                agent_id = f"agent_{i}_{j}"
                analysis = await get_structured_output(
                    llm=self.llm,
                    provider_config=self.llm_config,
                    prompt_template=prompt_service.get_template("attribute_and_hard_request_generator"),
                    input_data={"agent_prompt": agent_prompt, "vector_word_size": word_vector_size},
                    pydantic_schema=AgentAnalysisOutput
                )
                
                if not analysis:
                    analysis = AgentAnalysisOutput(attributes="", hard_request="Solve the original problem.")
                
                sub_problem = decomposed_problems_map.get(agent_id, user_prompt)
                
                assigned_mbti = random.choice(mbti_archetypes)
                assigned_name = agent_name_list.pop() if agent_name_list else names.get_full_name()
                self.agent_personas[agent_id] = {"mbti_type": assigned_mbti, "name": assigned_name}
                logging.info(f"LOG: Assigned Persona {assigned_name} (MBTI: {assigned_mbti}) to hidden agent {agent_id}")

                new_prompt = await dense_spanner_chain.ainvoke({
                    "attributes": analysis.attributes,
                    "hard_request": analysis.hard_request,
                    "sub_problem": sub_problem,
                    "mbti_type": assigned_mbti,
                    "name": assigned_name
                })
                current_layer_prompts.append(new_prompt)
            all_layers_prompts.append(current_layer_prompts)
            
        return all_layers_prompts, self.agent_personas

    async def _generate_seed_verbs(self, user_prompt: str, mbti_archetypes: List[str], word_vector_size: int) -> Dict[str, str]:
        """Generates a set of seed verbs based on the user's prompt."""
        num_agents_per_layer = len(mbti_archetypes)
        total_verbs_to_generate = word_vector_size * num_agents_per_layer
        seed_chain = prompt_service.create_chain(self.llm, "seed_generation", word_count=total_verbs_to_generate)
        
        verbs_str = await seed_chain.ainvoke({"problem": user_prompt})
        all_verbs = list(set(verbs_str.split()))
        
        if len(all_verbs) < total_verbs_to_generate:
            raise ValueError(f"CRITICAL: Seed verb generation failed. LLM produced only {len(all_verbs)}/{total_verbs_to_generate} required verbs. Halting.")
        
        random.shuffle(all_verbs)
        
        verb_chunks = [all_verbs[i:i + word_vector_size] for i in range(0, len(mbti_archetypes) * word_vector_size, word_vector_size)]
        return {mbti: " ".join(verb_chunks[i]) for i, mbti in enumerate(mbti_archetypes)}