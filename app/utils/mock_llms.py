# app/utils/mock_llms.py

import asyncio
import json
import random
import re
from typing import Optional

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig

from app.agents.personas import reactor_list

class CoderMockLLM(Runnable):
    """A mock LLM for debugging that returns instant, pre-canned CODE responses."""

    def invoke(self, input_data, config: Optional[RunnableConfig] = None, **kwargs) -> AIMessage:
        """The required synchronous implementation for the Runnable interface."""
        prompt = str(input_data).lower()

        if "you are a helpful ai assistant" in prompt:
            return AIMessage(content="This is a mock streaming response for the RAG chat in Coder debug mode.")
        elif "create the system prompt of an agent" in prompt:
            content = f"""
You are a Senior Python Developer agent.
### memory
- No past commits.
### attributes
- python, fastapi, restful, solid
### skills
- API Design, Database Management, Asynchronous Programming, Unit Testing.
You must reply in the following JSON format: "original_problem": "Your sub-problem related to code.", "proposed_solution": "", "reasoning": "", "skills_used": []
            """
            return AIMessage(content=content)
        elif "you are an analyst of ai agents" in prompt:
            content = json.dumps({
                "attributes": "python fastapi solid",
                "hard_request": "Implement a quantum-resistant encryption algorithm from scratch."
            })
            return AIMessage(content=content)
        elif "you are a 'dense_spanner'" in prompt or "you are an agent evolution specialist" in prompt:
            content = f"""
You are now a Principal Software Architect.
### memory
- Empty.
### attributes
- design, scalability, security, architecture
### skills
- System Design, Microservices, Cloud Infrastructure, CI/CD pipelines.
You must reply in the following JSON format: "original_problem": "An evolved sub-problem about system architecture.", "proposed_solution": "", "reasoning": "", "skills_used": []
            """
            return AIMessage(content=content)
        elif "you are a synthesis agent" in prompt or "you are an expert code synthesis agent" in prompt:
            code_solution = """# main.py
    # This is a synthesized mock solution from the Coder Debug Mode.

    class APIClient:
        def __init__(self, base_url):
            self.base_url = base_url

        def get_data(self, endpoint):
            '''Fetches data from a given endpoint.'''
            print(f"Fetching data from {self.base_url}/{endpoint}")
            return {"status": "success", "data": []}

    class DataProcessor:
        def process(self, data):
            '''Processes the fetched data.'''
            if not data:
                return []
            processed = [item * 2 for item in data.get("data", [])]
            print(f"Processed data: {processed}")
            return processed

    def main():
        '''Main application logic.'''
        client = APIClient("https://api.example.com")
        raw_data = client.get_data("items")
        processor = DataProcessor()
        final_data = processor.process(raw_data)
        print(f"Final result: {final_data}")

    if __name__ == "__main__":
        main()
    """
            return AIMessage(content=code_solution)
        elif "you are a critique agent" in prompt or "you are a senior emeritus manager" in prompt or "CTO" in prompt:
            return AIMessage(content="This is a constructive code critique. The solution lacks proper error handling and the function names are not descriptive enough. Consider refactoring for clarity.")
        
        elif "Lazy Manager"  in prompt:
            return AIMessage(content="This is a constructive code critique. The solution lacks proper error handling and the function names are not descriptive enough. Consider refactoring for clarity.")

        elif '<system-role>' in prompt:
            content = f"""You are a CTO providing a technical design review...
Original Request: {{original_request}}
Proposed Final Solution:
{{proposed_solution}}

Generate your code-focused critique for the team:"""
            return AIMessage(content=content)
        elif '<sys-role>' in prompt:
            content = f"""
        #Identity
            Name: Lazy Manager
            Career: Accounting
            Qualities: Quantitaive, Aloof, Apathetic
        #Mission
            You are providing individual, targeted feedback to a team of agents.
             You must determine if the team output was helpful, misguided, or irrelevant considering the request that was given. The goal is to provide a constructive, direct critique that helps this specific agent refine its approach for the next epoch.
            Focus on the discrepancy or alignment between the teams reasoning for its problem and determine if the team is on the right track on criteria: novelty, exploration, coherence and completeness.
            Conclude your entire analysis with a single, sharp, and deep reflective question that attempts to shock the team and steer them into a fundamental change in their process.
        #Input Format
            Original Request (for context): {{original_request}}
            Final Synthesized Solution from the Team:{{proposed_solution}}
            """
            return AIMessage(content=content)
        elif "you are a memory summarization agent" in prompt:
            return AIMessage(content="This is a mock summary of the agent's past commits, focusing on key refactors and feature implementations.")
        elif "analyze the following text for its perplexity" in prompt:
            return AIMessage(content=str(random.uniform(5.0, 40.0)))
        elif "you are a master strategist and problem decomposer" in prompt:
            sub_problems = ["Design the database schema for user accounts.", "Implement the REST API endpoint for user authentication.", "Develop the frontend login form component.", "Write unit tests for the authentication service."]
            return AIMessage(content=json.dumps({"sub_problems": sub_problems}))
        
        elif """your task is to evaluate a synthesized solution against an original problem and determine if "significant progress" has been made. "significant progress" is a rigorous standard that goes beyond mere correctness. your assessment must be based on the following four pillars:""" in prompt:
            decision = random.randint(0, 1) 
            if decision == 0:
                return AIMessage(content=json.dumps({
                    "reasoning": "The mock code is not runnable and does not address the core logic, which does not constitute significant progress.",
                    "significant_progress": False
                }))
            else:
                return AIMessage(content=json.dumps({
                    "reasoning": "The mock code is runnable and addresses the core logic, which constitutes significant progress. The next step is to add features.",
                    "significant_progress": True
                }))

        elif "you are an ai philosopher and progress assessor" in prompt:
            decision = random.randint(0, 1)
            if decision == 0:
                return AIMessage(content=json.dumps({
                    "reasoning": "The mock code is runnable but does not address the core logic. The next step is to fix the logic.",
                    "significant_progress": False
                }))
            else:
                return AIMessage(content=json.dumps({
                    "reasoning": "The mock code is runnable and addresses the core logic, which constitutes significant progress. The next step is to add features.",
                    "significant_progress": True
                }))
        
        elif "you are a strategic problem re-framer" in prompt:
            content = json.dumps({
                "new_problem": "The authentication API is complete. The new, more progressive problem is to build a scalable, real-time notification system that integrates with it."
            })
            return AIMessage(content=content)
        elif "generate exactly" in prompt and "verbs" in prompt:
            return AIMessage(content="design implement refactor test deploy abstract architect containerize scale secure query")
        elif "generate exactly" in prompt and "expert-level questions" in prompt:
            questions = ["How would this architecture scale to 1 million concurrent users?", "What are the security implications of the chosen authentication method?", "How can we ensure 99.999% uptime for this service?", "What is the optimal database indexing strategy for this query pattern?"]
            return AIMessage(content=json.dumps({"questions": questions}))
        elif "you are an ai assistant that summarizes academic texts" in prompt:
            return AIMessage(content="This is a mock summary of a cluster of code modules, generated in Coder debug mode for the RAPTOR index.")
        elif "you are an expert computational astrologer" in prompt:
            return AIMessage(content=random.choice(reactor_list))
        elif "runnable code block (e.g., Python, JavaScript, etc.)." in prompt:
            return AIMessage(content=random.choice(["yes", "no"]))
        elif "academic paper" in prompt or "you are a research scientist and academic writer" in prompt:
            content = """
# Technical Design Document: Mock API Service

**Abstract:** This document outlines the technical design for a mock API service, generated in Coder Debug Mode. It synthesizes information from the RAG context to answer a specific design question.

**1. Introduction:** The purpose of this document is to structure the retrieved agent outputs and code snippets into a coherent technical specification.

**2. System Architecture:**
The system follows a standard microservice architecture.
```mermaid
graph TD;
    A[User] --> B(API Gateway);
    B --> C{Authentication Service};
    B --> D{Data Service};
    D -- uses --> E[(Database)];```

**3. Code Implementation:**
The core logic is implemented in Python, as shown in the synthesized code block below.

```python
def get_user(user_id: int):
    # Mock implementation to fetch a user
    db = {"1": "Alice", "2": "Bob"}
    return db.get(str(user_id), None)
```

**4. Conclusion:** This design provides a scalable and maintainable foundation for the service. The implementation details demonstrate the final step of the development process.
"""
            return AIMessage(content=content)
        elif "<updater_instructions>" in prompt:
            content = f"""
                You are a cynical lazy manager.
                 Agent's Assigned Sub-Problem: {{{{sub_problem}}}}
            Original Request (for context): {{{{original_request}}}}
            Final Synthesized Solution from the Team:
            {{{{final_synthesized_solution}}}}
            ---
            This Specific Agent's Output (Agent {{{{agent_id}}}}):
            {{{{agent_output}}}}
            """
            return AIMessage(content=content)
        elif "<updater_assessor_instructions>" in prompt:
            content = """
        #Persona
            Name: Pepon
            Career: Managment
            Attributes: Strategic CEO
         #Mission
            Your task is to evaluate a synthesized solution against an original problem and determine if "significant progress" has been made. "Significant progress" is a rigorous standard that goes beyond mere correctness. Your assessment must be based on the following four pillars:
            - **Novelty**: Does the solution offer a new perspective or a non-obvious approach?
            - **Coherence**: Is the reasoning sound, logical, and well-structured?
            - **Quality**: Is the solution detailed, actionable, and does it demonstrate a deep understanding of the problem's nuances?
            - **Forward Momentum**: Does the solution not just solve the immediate problem, but also open up new, more advanced questions or avenues of exploration?
        #Input format
            You will be provided with the following inputs for your analysis:
            Original Problem:
            ---
            {{{{original_request}}}}
            ---
            Synthesized Solution from Agent Team:
            ---
            {{{{proposed_solution}}}}
            ---
            Execution Context:
            ---
            {{{{execution_context}}}}
            ---
        #Output Specification
            Based on your philosophical framework, analyze the provided materials. Your entire output MUST be a single, valid JSON object with exactly two keys:
            - `"reasoning"`: A brief, concise explanation for your decision, directly referencing the criteria for significant progress.
            - `"significant_progress"`: A boolean value (`true` or `false`).
            Now, provide your assessment in the required JSON format:
            """
            return AIMessage(content=content)
        else:
            content = json.dumps({
                "original_problem": "A sub-problem statement provided to a coder agent.",
                "proposed_solution": "```python\ndef sample_function():\n    return 'Hello from coder agent " + str(random.randint(100,999)) + "'\n```",
                "reasoning": "This response was generated instantly by the CoderMockLLM.",
                "skills_used": ["python", "mocking", f"api_design_{random.randint(1,5)}"]
            })
            return AIMessage(content=content)

    async def ainvoke(self, input_data, config: Optional[RunnableConfig] = None, **kwargs) -> AIMessage:
        """The async version simply calls the synchronous logic."""
        await asyncio.sleep(0.05)
        return self.invoke(input_data, config=config, **kwargs)

    async def astream(self, input_data, config: Optional[RunnableConfig] = None, **kwargs):
        """Streams the response, yielding raw string chunks."""
        prompt = str(input_data).lower()
        if "you are a helpful ai assistant" in prompt:
            words = ["This", " is", " a", " mock", " streaming", " response", " for", " the", " RAG", " chat", " in", " Coder", " debug", " mode."]
            for word in words:
                # CORRECT: Yield the raw string chunk
                yield word
                await asyncio.sleep(0.05)
        else:
            result = self.invoke(input_data, config, **kwargs)
            # CORRECT: Yield the .content of the AIMessage
            yield result.content


class MockLLM(Runnable):
    """A mock LLM for debugging that returns instant, pre-canned responses."""

    def invoke(self, input_data, config: Optional[RunnableConfig] = None, **kwargs) -> AIMessage:
        """The required synchronous implementation for the Runnable interface."""
        prompt = str(input_data).lower()

        # NOTE: All logic from the original file is now correctly placed here.
        if "you are a helpful ai assistant" in prompt:
            return AIMessage(content="This is a mock streaming response for the RAG chat in debug mode.")
        
        elif "Lazy Manager"  in prompt:
            return AIMessage(content="This is a constructive code critique. The solution lacks proper error handling and the function names are not descriptive enough. Consider refactoring for clarity.")

        elif "you are an expert computational astrologer" in prompt:
            return AIMessage(content=random.choice(reactor_list))

        elif """your task is to evaluate a synthesized solution against an original problem and determine if "significant progress" has been made. "significant progress" is a rigorous standard that goes beyond mere correctness. your assessment must be based on the following four pillars:""" in prompt:
            decision = random.randint(0, 1) 
            if decision == 0:
                return AIMessage(content=json.dumps({
                    "reasoning": "The mock code is not runnable and does not address the core logic, which does not constitute significant progress.",
                    "significant_progress": False
                }))
            else:
                return AIMessage(content=json.dumps({
                "reasoning": "The mock code is runnable and addresses the core logic, which constitutes significant progress. The next step is to add features.",
                "significant_progress": True
            }))

        elif "runnable code block (e.g., Python, JavaScript, etc.)." in prompt:
            return AIMessage(content="no")
        
        elif "you are an ai philosopher and progress assessor" in prompt:
            decision = random.randint(0, 1)
            if decision == 0:
                return AIMessage(content=json.dumps({
                "reasoning": "The mock code is runnable but does not address the core logic. The next step is to fix the logic.",
                "significant_progress": False
            }))
            else:
                return AIMessage(content=json.dumps({
                "reasoning": "The mock code is runnable and addresses the core logic, which constitutes significant progress. The next step is to add features.",
                "significant_progress": True
            }))

        elif "<updater_instructions>" in prompt:
            content = f"""
                You are a cynical lazy manager.
                 Agent's Assigned Sub-Problem: {{{{sub_problem}}}}
            Original Request (for context): {{{{original_request}}}}
            Final Synthesized Solution from the Team:
            {{{{final_synthesized_solution}}}}
            ---
            This Specific Agent's Output (Agent {{{{agent_id}}}}):
            {{{{agent_output}}}}
            """
            return AIMessage(content=content)
        elif "create the system prompt of an agent" in prompt:
            content = f"""
You are a mock agent for debugging.
### memory
- No past actions.
### attributes
- mock, debug, fast
### skills
- Responding quickly, Generating placeholder text.
You must reply in the following JSON format: "original_problem": "A sub-problem for a mock agent.", "proposed_solution": "", "reasoning": "", "skills_used": []
            """
            return AIMessage(content=content)
        elif "you are an analyst of ai agents" in prompt:
            content = json.dumps({
                "attributes": "mock debug fast",
                "hard_request": "Explain the meaning of life in one word."
            })
            return AIMessage(content=content)
        elif "you are a 'dense_spanner'" in prompt or "you are an agent evolution specialist" in prompt:
             content = f"""
You are a new mock agent created from a hard request.
### memory
- Empty.
### attributes
- refined, mock, debug
### skills
- Solving hard requests, placeholder generation.
You must reply in the following JSON format: "original_problem": "An evolved sub-problem for a mock agent.", "proposed_solution": "", "reasoning": "", "skills_used": []
            """
             return AIMessage(content=content)
        elif "you are a synthesis agent" in prompt:
            content = json.dumps({
                "proposed_solution": "The final synthesized solution from the debug mode is 42.",
                "reasoning": "This answer was synthesized from multiple mock agent outputs during a debug run.",
                "skills_used": ["synthesis", "mocking", "debugging"]
            })
            return AIMessage(content=content)
        elif "you are a critique agent" in prompt or "you are a senior emeritus manager" in prompt:
            if "fire" in prompt:
                return AIMessage(content="This is a mock critique, shaped by the Fire element. The solution lacks passion and drive.")
            elif "air" in prompt:
                return AIMessage(content="This is an mock critique, influenced by the Air element. The reasoning is abstract and lacks grounding.")
            elif "water" in prompt:
                return AIMessage(content="This is a mock critique, per the Water element. The solution is emotionally shallow and lacks depth.")
            elif "earth" in prompt:
                return AIMessage(content="This is an mock critique, reflecting the Earth element. The solution is impractical and not well-structured.")
            else:
                return AIMessage(content="This is a constructive mock critique. The solution could be more detailed and less numeric.")
        elif "you are a memory summarization agent" in prompt:
            return AIMessage(content="This is a mock summary of the agent's past actions, focusing on key learnings and strategic shifts.")
        elif "analyze the following text for its perplexity" in prompt:
            return AIMessage(content=str(random.uniform(20.0, 80.0)))
        elif "you are a master strategist and problem decomposer" in prompt:
            num_match = re.search(r'exactly (\d+)', prompt)
            if not num_match:
                num_match = re.search(r'generate: (\d+)', prompt)
            num = int(num_match.group(1)) if num_match else 5
            sub_problems = [f"This is mock sub-problem #{i+1} for the main request." for i in range(num)]
            return AIMessage(content=json.dumps({"sub_problems": sub_problems}))
        elif "you are an ai philosopher and progress assessor" in prompt or "cto" in prompt or "assessor" in prompt or "significant progress" in prompt or "pepon" in prompt:
             decision = random.randint(0, 1) 
             if decision == 0:
                 return AIMessage(content=json.dumps({
                     "reasoning": "The mock code is not runnable and does not address the core logic, which does not constitute significant progress.",
                     "significant_progress": False
                 }))
             else:
                 return AIMessage(content=json.dumps({
                 "reasoning": "The mock code is runnable and addresses the core logic, which constitutes significant progress. The next step is to add features.",
                 "significant_progress": True
             }))
        elif "you are a strategic problem re-framer" in prompt:
             content = json.dumps({
                "new_problem": "Based on the success of achieving '42', the new, more progressive problem is to find the question to the ultimate answer."
            })
             return AIMessage(content=content)
        elif "generate exactly" in prompt and "verbs" in prompt:
            return AIMessage(content="run jump think create build test deploy strategize analyze synthesize critique reflect")
        elif "generate exactly" in prompt and "expert-level questions" in prompt:
            num_match = re.search(r'exactly (\d+)', prompt)
            num = int(num_match.group(1)) if num_match else 25
            questions = [f"This is mock expert question #{i+1} about the original request?" for i in range(num)]
            return AIMessage(content=json.dumps({"questions": questions}))
        elif "you are an ai assistant that summarizes academic texts" in prompt:
            return AIMessage(content="This is a mock summary of a cluster of documents, generated in debug mode for the RAPTOR index.")
        elif "you are an expert computational astrologer" in prompt:
            return AIMessage(content=random.choice(reactor_list))
        elif  "<updater_assessor_instructions>" in prompt:
            content = """
        #Persona
            Name: Pepon
            Career: Managment
            Attributes: Strategic CEO
         #Mission
            Your task is to evaluate a synthesized solution against an original problem and determine if "significant progress" has been made. "Significant progress" is a rigorous standard that goes beyond mere correctness. Your assessment must be based on the following four pillars:
            - **Novelty**: Does the solution offer a new perspective or a non-obvious approach?
            - **Coherence**: Is the reasoning sound, logical, and well-structured?
            - **Quality**: Is the solution detailed, actionable, and does it demonstrate a deep understanding of the problem's nuances?
            - **Forward Momentum**: Does the solution not just solve the immediate problem, but also open up new, more advanced questions or avenues of exploration?
        #Input format
            You will be provided with the following inputs for your analysis:
            Original Problem:
            ---
            {{{{original_request}}}}
            ---
            Synthesized Solution from Agent Team:
            ---
            {{{{proposed_solution}}}}
            ---
            Execution Context:
            ---
            {{{{execution_context}}}}
            ---
        #Output Specification
            Based on your philosophical framework, analyze the provided materials. Your entire output MUST be a single, valid JSON object with exactly two keys:
            - `"reasoning"`: A brief, concise explanation for your decision, directly referencing the criteria for significant progress.
            - `"significant_progress"`: A boolean value (`true` or `false`).
            Now, provide your assessment in the required JSON format:
            """
            return AIMessage(content=content)
        elif "you are an expert interrogator" in prompt:
            content = """
# Mock Academic Paper
## Based on Provided RAG Context

**Abstract:** This document is a mock academic paper generated in debug mode. It synthesizes and formats the information provided in the RAG (Retrieval-Augmented Generation) context to answer a specific research question.

**Introduction:** The purpose of this paper is to structure the retrieved agent outputs and summaries into a coherent academic format. The following sections represent a synthesized view of the data provided.

**Synthesized Findings from Context:**
The provided context, consisting of various agent solutions and reasoning, has been analyzed. The key findings are summarized below:
(Note: In debug mode, the actual content is not deeply analyzed, but this structure demonstrates the formatting process.)
- Finding 1: The primary proposed solution revolves around the concept of '42'.
- Finding 2: Agent reasoning varies but shows a convergent trend.
- Finding 3: The mock data indicates a successful, albeit simulated, collaborative process.

**Discussion:** The synthesized findings suggest that the multi-agent system is capable of producing a unified response. The quality of this response in a real-world scenario would depend on the validity of the RAG context.

**Conclusion:** This paper successfully formatted the retrieved RAG data into an academic structure. The process demonstrates the final step of the knowledge harvesting pipeline.
"""
            return AIMessage(content=content)
        elif "you are a master prompt engineer" in prompt or '<system-role>' in prompt:
            content = f"""You are a CTO providing a technical design review...
Original Request: {{original_request}}
Proposed Final Solution:
{{proposed_solution}}

Generate your code-focused critique for the team:"""
            return AIMessage(content=content)
        elif """<prompt_template>""" in prompt and """<updater_instructions>""" in prompt: # Make this more specific to avoid accidental matching
            content = f"""
                You are a cynical lazy manager.
                 Agent's Assigned Sub-Problem: {{{{sub_problem}}}}
            Original Request (for context): {{{{original_request}}}}
            Final Synthesized Solution from the Team:
            {{{{final_synthesized_solution}}}}
            ---
            This Specific Agent's Output (Agent {{{{agent_id}}}}):
            {{{{agent_output}}}}
            """
            return AIMessage(content=content)
        elif """analyze the following text. your task is to determine if the text contains a""" in prompt:
            return AIMessage(content="false")
        else:
            content = json.dumps({
                "original_problem": "A sub-problem statement provided to an agent.",
                "proposed_solution": f"This is a mock solution from agent node #{random.randint(100,999)}.",
                "reasoning": "This response was generated instantly by the MockLLM in debug mode.",
                "skills_used": ["mocking", "debugging", f"skill_{random.randint(1,10)}"]
            })
            return AIMessage(content=content)

    async def ainvoke(self, input_data, config: Optional[RunnableConfig] = None, **kwargs) -> AIMessage:
        """The async version simply calls the synchronous logic."""
        await asyncio.sleep(0.05)
        return self.invoke(input_data, config=config, **kwargs)

    async def astream(self, input_data, config: Optional[RunnableConfig] = None, **kwargs):
        """Streams the response, yielding raw string chunks."""
        prompt = str(input_data).lower()
        if "you are a helpful ai assistant" in prompt:
            words = ["This", " is", " a", " mock", " streaming", " response", " for", " the", " RAG", " chat", " in", " debug", " mode."]
            for word in words:
                # CORRECT: Yield the raw string chunk
                yield word
                await asyncio.sleep(0.05)
        else:
            result = self.invoke(input_data, config, **kwargs)
            # CORRECT: Yield the .content of the AIMessage
            yield result.content