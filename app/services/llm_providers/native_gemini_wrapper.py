# app/services/llm_providers/native_gemini_wrapper.py

from google import genai
from google.genai import types
from typing import Any, List
from langchain_core.callbacks.manager import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

def _convert_to_google_genai_contents(messages: List[BaseMessage]) -> tuple[List[types.Content], str | None]:
    """
    Converts LangChain messages to the new `google-genai` Content block structure
    and extracts the system prompt, as per the new SDK guidelines.
    """
    contents = []
    system_prompt = None
    for message in messages:
        if isinstance(message, SystemMessage):
            system_prompt = message.content
        elif isinstance(message, HumanMessage):
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text=message.content)]))
        elif isinstance(message, AIMessage):
            contents.append(types.Content(role="model", parts=[types.Part.from_text(text=message.content)]))
        else:
            raise ValueError(f"Unsupported message type: {type(message)}")
    return contents, system_prompt

class NativeGeminiChatWrapper(BaseChatModel):
    """
    A LangChain ChatModel wrapper for the new native `google-genai` library.
    This version correctly handles structured output requests and provides both sync and async methods.
    """
    client: genai.Client
    model_name: str
    model_kwargs: dict = {}
    
    _is_structured_output_supported: bool = True
    
    @property
    def _llm_type(self) -> str:
        return "native-gemini-chat-sdk-v2-structured-output"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: List[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """The synchronous implementation of the chat model call."""
        # ... function content is unchanged ...
        contents, system_prompt = _convert_to_google_genai_contents(messages)

        safety_settings = [
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        ]
        
        config_args = self.model_kwargs.copy()
        if system_prompt:
            config_args['system_instruction'] = system_prompt

        if "response_schemas" in kwargs and kwargs["response_schemas"]:
            pydantic_schema = kwargs["response_schemas"][0]
            config_args['response_mime_type'] = "application/json"
            config_args['response_schema'] = pydantic_schema
            
        config = types.GenerateContentConfig(**config_args, safety_settings=safety_settings)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )
        
        text_response = response.text
        message = AIMessage(content=text_response)
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: List[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """The asynchronous implementation of the chat model call."""
        # ... function content is unchanged ...
        contents, system_prompt = _convert_to_google_genai_contents(messages)

        safety_settings = [
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        ]
        
        config_args = self.model_kwargs.copy()
        if system_prompt:
            config_args['system_instruction'] = system_prompt

        if "response_schemas" in kwargs and kwargs["response_schemas"]:
            pydantic_schema = kwargs["response_schemas"][0]
            config_args['response_mime_type'] = "application/json"
            config_args['response_schema'] = pydantic_schema
            
        config = types.GenerateContentConfig(**config_args, safety_settings=safety_settings)

        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )
        
        text_response = response.text
        message = AIMessage(content=text_response)
        return ChatResult(generations=[ChatGeneration(message=message)])