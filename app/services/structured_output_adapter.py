# app/services/structured_output_adapter.py

import asyncio
import json
import logging
import re
from typing import Type, Optional
from pydantic import BaseModel, ValidationError
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.core.config import ProviderConfig

def _parse_delimiter_output(text: str, pydantic_schema: Type[BaseModel]) -> dict:
    """Helper function to parse text with custom delimiters into a dictionary."""
    parsed_data = {}
    fields = pydantic_schema.model_fields
    for field_name in fields.keys():
        tag = field_name.upper()
        # Non-greedy match for content between tags or until the end of the string
        pattern = rf"\[---{tag}---\]([\s\S]*?)(\[---|\Z)"
        match = re.search(pattern, text)
        if match:
            value = match.group(1).strip()
            # Handle specific type conversions that simple parsing might miss
            field_info = fields[field_name]
            if "bool" in str(field_info.annotation):
                parsed_data[field_name] = value.lower() in ['true', '1', 't', 'y', 'yes']
            elif "List" in str(field_info.annotation):
                 parsed_data[field_name] = [item.strip() for item in value.split(',') if item.strip()]
            else:
                 parsed_data[field_name] = value
        else:
            logging.warning(f"Delimiter parsing: Did not find tag for field '{field_name}'.")
    return parsed_data

async def get_structured_output(
    llm: BaseChatModel,
    provider_config: ProviderConfig,
    prompt_template: str,
    input_data: dict,
    pydantic_schema: Type[BaseModel],
    max_retries: int = 2
) -> Optional[BaseModel]:
    """
    This function is the robust adapter for Structured Data Endpoints.
    It tries the best method first and falls back to a dynamically generated,
    universal delimiter method. It also includes a retry mechanism.
    """
    last_exception = None
    for attempt in range(max_retries):
        # --- STRATEGY 1: Attempt Native Tool Calling / Structured Output ---
        if provider_config.enable_structured_output:
            try:
                logging.info(f"Attempting structured output via native tool calling for {pydantic_schema.__name__} (Attempt {attempt+1}/{max_retries}).")
                structured_llm = llm.with_structured_output(pydantic_schema)
                prompt = ChatPromptTemplate.from_template(prompt_template)
                chain = prompt | structured_llm
                result = await chain.ainvoke(input_data)
                if result:
                     return result # Success!
                # If result is None, fall through to the next strategy
                logging.warning(f"Native tool calling returned None for {pydantic_schema.__name__}. Falling back.")
            except Exception as e:
                logging.warning(f"Native tool calling failed (Attempt {attempt+1}/{max_retries}): {e}. Falling back to delimiter method.")
                last_exception = e

        # --- STRATEGY 2: Fallback to Dynamic Structured Delimiters ---
        response_str = None
        full_prompt_str = ""
        try:
            logging.info(f"Attempting structured output via dynamic delimiters for {pydantic_schema.__name__} (Attempt {attempt+1}/{max_retries}).")
            
            # Delimiter instruction building logic
            fields = pydantic_schema.model_fields
            delimiter_instructions = []
            for field_name, field_info in fields.items():
                tag = field_name.upper()
                type_hint_str = str(field_info.annotation)
                if "List" in type_hint_str:
                    # Assign the multi-line string to the 'type_hint' variable.
                    type_hint = (
                        "a comma-separated list of strings. "
                        "For example: `item1, item2, item3` "
                        "DO NOT use parentheses `()`, square brackets `[]`, or trailing punctuation."
                    )
                elif "bool" in type_hint_str:
                    type_hint = "either 'true' or 'false'"
                else:
                    type_hint = "a string value"

                description = field_info.description or 'No description.'
                delimiter_instructions.append(f"[---{tag}---]\n({description} Your value should be {type_hint}.)")

            delimiter_instructions_str = "\n\n".join(delimiter_instructions)

            base_prompt_template = ChatPromptTemplate.from_template(prompt_template)
            # We must convert the ChatPromptValue to a string before using it in the next f-string.
            formatted_base_prompt_obj = await base_prompt_template.ainvoke(input_data)
            formatted_base_prompt_str = formatted_base_prompt_obj.to_string()

            full_prompt_str = f"""{formatted_base_prompt_str}
---
Output Mandate:
You MUST format your response using ONLY the following structure. Do not add any other text, explanation, or markdown.

{delimiter_instructions_str}
"""

            # This is the definitive log of the exact prompt sent to the LLM.
            logging.info(f"RAW_LLM_INPUT_FOR_DEBUG ({pydantic_schema.__name__}):\n---BEGIN INPUT---\n{full_prompt_str}\n---END INPUT---")
            
            final_prompt = ChatPromptTemplate.from_template("{final_prompt_str}")
            chain = final_prompt | llm | StrOutputParser()
            response_str = await chain.ainvoke({"final_prompt_str": full_prompt_str})

            # THE 'BLACK BOX' LOGGING IS INSERTED *BEFORE* ANY PARSING ATTEMPT.
            # THIS ENSURES WE CAPTURE THE RAW DATA REGARDLESS OF THE FAILURE MODE.
            logging.info(f"RAW_LLM_OUTPUT_FOR_DEBUG ({pydantic_schema.__name__}):\n---BEGIN RAW---\n{response_str}\n---END RAW---")

            parsed_data = _parse_delimiter_output(response_str, pydantic_schema)
            logging.info(f"Pre-validation data for {pydantic_schema.__name__}: {parsed_data}")
            validated_output = pydantic_schema.model_validate(parsed_data)
            return validated_output

        except (ValidationError, Exception) as e:
            # ON FAILURE, THE LOG NOW INCLUDES THE RAW RESPONSE STRING THAT CAUSED THE EXCEPTION.
            error_message = f"Delimiter-based structured output failed on attempt {attempt+1}/{max_retries}. Error: {e}"
            logging.error(
                f"{error_message}\n"
                f"--- FAILED RAW PROMPT ---\n{full_prompt_str}\n"
                f"--- FAILED RAW LLM OUTPUT ---\n{response_str}\n"
                f"--- END OF FAILURE LOG ---"
            )
            last_exception = e
            if attempt < max_retries - 1:
                await asyncio.sleep(1)

    logging.error(f"All {max_retries} attempts to get structured output for {pydantic_schema.__name__} failed. Last error: {last_exception}")
    return None