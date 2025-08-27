# app/services/prompt_service.py

from pathlib import Path
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import Runnable

class PromptService:
    """
    Manages loading prompt templates and creating LangChain runnable chains.
    This version uses LangChain's native .partial() method for pre-formatting,
    which correctly handles escaped braces in templates.
    """
    def __init__(self, prompts_dir: str = "app/prompts"):
        self.prompts_path = Path(prompts_dir)
        self.templates = self._load_templates()

    def _load_templates(self) -> dict[str, str]:
        """Loads all .prompt files from the prompts directory."""
        templates = {}
        for file_path in self.prompts_path.glob("*.prompt"):
            with open(file_path, "r", encoding="utf-8") as f:
                templates[file_path.stem] = f.read()
        return templates

    def get_template(self, name: str) -> str:
        """Gets a raw, unformatted prompt template string."""
        if name not in self.templates:
            # A special case for the dynamically generated prompt in critique node
            #  a special key for in-memory templates used by the reflection nodes
            if name == "raw_string_template":
                 return "{raw_template_str}" # Placeholder
            raise ValueError(f"Prompt template '{name}' not found.")
        return self.templates[name]

    def create_chain(self, llm, prompt_name: str, is_chat_prompt: bool = False, **kwargs) -> Runnable:
        """
        Creates a LangChain runnable from a prompt template.
        
        Args:
            llm: The language model to use.
            prompt_name: The name of the prompt file (without extension).
            is_chat_prompt: Set to True for chains that don't need a StrOutputParser.
            **kwargs: Arguments to pre-format (partially fill) the prompt template.
        """
        if prompt_name == "raw_string_template" and "raw_template_str" in kwargs:
             template_str = kwargs.pop("raw_template_str")
        else:
             template_str = self.get_template(prompt_name)
        
        prompt = ChatPromptTemplate.from_template(template_str)
        
        if kwargs:
            prompt = prompt.partial(**kwargs)
        
        if is_chat_prompt:
            return prompt | llm
        
        return prompt | llm | StrOutputParser()

# Global instance
prompt_service = PromptService()