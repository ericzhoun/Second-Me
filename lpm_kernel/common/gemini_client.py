"""
Google Gemini API client adapter that provides an OpenAI-compatible interface.
Uses the official google-generativeai Python SDK.
"""

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import logging
import time
import uuid
from typing import List, Dict, Any, Optional, Iterator

logger = logging.getLogger(__name__)

# Default Gemini model to use if an invalid model is provided
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"


class GeminiClient:
    """
    Adapter for Google Gemini API to mimic OpenAI client interface.
    Uses the official google-generativeai SDK.
    """
    
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        """
        Initialize the Gemini client.
        
        Args:
            api_key: Google AI API key
            base_url: Optional base URL (not used by Gemini SDK, kept for compatibility)
        """
        if not api_key:
            raise ValueError("API key is required for GeminiClient")
        
        # Configure the Gemini API with the provided key
        genai.configure(api_key=api_key)
        self.api_key = api_key
        self.base_url = base_url
        
        # Create the chat interface to mimic OpenAI structure
        self.chat = self._ChatNamespace(self)
        
        # Safety settings - set to minimum blocking for flexibility
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

    class _ChatNamespace:
        """Namespace class to mimic OpenAI's client.chat structure."""
        
        def __init__(self, client: 'GeminiClient'):
            self.client = client
            self.completions = self._CompletionsNamespace(client)

        class _CompletionsNamespace:
            """Namespace class to mimic OpenAI's client.chat.completions structure."""
            
            def __init__(self, client: 'GeminiClient'):
                self.client = client

            def create(self, **kwargs) -> Any:
                """Create a chat completion, mimicking OpenAI's interface."""
                return self.client._create_completion(**kwargs)

    def _validate_model_name(self, model: str) -> str:
        """
        Validate and normalize the model name for Gemini API.
        
        Args:
            model: The model name provided
            
        Returns:
            A valid Gemini model name
        """
        if not model:
            logger.warning(f"No model specified, using default: {DEFAULT_GEMINI_MODEL}")
            return DEFAULT_GEMINI_MODEL
        
        # Check if it's a valid Gemini model name
        model_lower = model.lower()
        if "gemini" not in model_lower:
            logger.warning(f"Non-Gemini model name '{model}' provided. Using default: {DEFAULT_GEMINI_MODEL}")
            return DEFAULT_GEMINI_MODEL
        
        # Remove 'models/' prefix if present (SDK adds it automatically)
        if model.startswith("models/"):
            model = model[7:]
        
        return model

    def _convert_messages_to_contents(self, messages: List[Dict[str, str]]) -> tuple:
        """
        Convert OpenAI-format messages to Gemini format.
        
        Args:
            messages: List of messages in OpenAI format
            
        Returns:
            Tuple of (contents_list, system_instruction)
        """
        system_instruction = None
        contents = []
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "system":
                # Accumulate system messages into system_instruction
                if system_instruction:
                    system_instruction += "\n" + content
                else:
                    system_instruction = content
            elif role == "user":
                contents.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [content]})
        
        return contents, system_instruction

    def _create_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> Any:
        """
        Create a chat completion using Gemini API.
        
        Args:
            messages: List of messages in OpenAI format
            model: Model name to use
            stream: Whether to stream the response
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            top_p: Top-p sampling parameter
            **kwargs: Additional parameters (ignored for compatibility)
            
        Returns:
            OpenAI-compatible response object
        """
        try:
            # Validate and normalize model name
            model = self._validate_model_name(model)
            
            # Convert messages to Gemini format
            contents, system_instruction = self._convert_messages_to_contents(messages)
            
            # Validate we have content to send
            if not contents:
                raise ValueError("No user/assistant messages provided")
            
            # Get the last user message as the prompt
            if contents[-1]["role"] == "user":
                prompt = contents[-1]["parts"][0]
                history = contents[:-1] if len(contents) > 1 else []
            else:
                # If last message is from model, we need a user prompt
                prompt = " "  # Placeholder
                history = contents
            
            # Validate prompt is not empty
            if not prompt or not str(prompt).strip():
                logger.warning("Empty prompt detected, using placeholder")
                prompt = "(No content provided)"
            
            # Build generation config
            generation_config = {
                "temperature": temperature,
            }
            if max_tokens:
                generation_config["max_output_tokens"] = max_tokens
            if top_p is not None and top_p > 0:
                generation_config["top_p"] = top_p
            
            # Create the model instance
            gemini_model = genai.GenerativeModel(
                model_name=model,
                system_instruction=system_instruction,
                generation_config=generation_config,
                safety_settings=self.safety_settings
            )
            
            # Start chat session with history
            chat = gemini_model.start_chat(history=history)
            
            # Send message
            if stream:
                response = chat.send_message(prompt, stream=True)
                return self._stream_response_adapter(response, model)
            else:
                response = chat.send_message(prompt)
                return self._response_adapter(response, model)
                
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            raise

    def _response_adapter(self, response, model: str) -> 'OpenAIResponse':
        """
        Adapt Gemini response to OpenAI format.
        
        Args:
            response: Gemini response object
            model: Model name used
            
        Returns:
            OpenAI-compatible response object
        """
        text = ""
        finish_reason = "stop"
        
        try:
            # Try to get text from response
            text = response.text
        except ValueError as e:
            # Content might be blocked
            logger.warning(f"Could not get response text: {str(e)}")
            finish_reason = "content_filter"
        
        # Log additional info if response is empty or blocked
        if not text:
            logger.warning("Gemini returned empty response")
            self._log_response_details(response)
        
        # Build OpenAI-compatible response
        return OpenAIResponse({
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": OpenAIMessage({
                    "role": "assistant",
                    "content": text or ""
                }),
                "finish_reason": finish_reason
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        })

    def _stream_response_adapter(self, response_iterator, model: str) -> Iterator['OpenAIResponse']:
        """
        Adapt Gemini streaming response to OpenAI format.
        
        Args:
            response_iterator: Gemini streaming response
            model: Model name used
            
        Yields:
            OpenAI-compatible streaming response chunks
        """
        response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        for chunk in response_iterator:
            try:
                text = chunk.text
                if text:
                    yield OpenAIResponse({
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": text},
                            "finish_reason": None
                        }]
                    })
            except ValueError:
                # Skip chunks that can't be read
                continue

        # Final chunk with finish reason
        yield OpenAIResponse({
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        })

    def _log_response_details(self, response) -> None:
        """Log detailed information about a Gemini response for debugging."""
        try:
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                logger.warning(f"Prompt feedback: {response.prompt_feedback}")
            
            if hasattr(response, 'candidates') and response.candidates:
                for i, candidate in enumerate(response.candidates):
                    finish_reason = getattr(candidate, 'finish_reason', 'unknown')
                    logger.warning(f"Candidate {i} finish_reason: {finish_reason}")
                    
                    if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                        for rating in candidate.safety_ratings:
                            logger.warning(f"  Safety: {rating.category.name} = {rating.probability.name}")
        except Exception as e:
            logger.debug(f"Could not log response details: {e}")

class OpenAIResponse(dict):
    """
    Helper class that allows both dict-style and attribute-style access.
    Mimics OpenAI response object structure.
    """
    
    def __init__(self, data: dict):
        super().__init__(data)
        for key, value in data.items():
            if isinstance(value, dict):
                self[key] = OpenAIResponse(value)
            elif isinstance(value, list):
                self[key] = [
                    OpenAIResponse(item) if isinstance(item, dict) else item
                    for item in value
                ]

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


class OpenAIMessage(OpenAIResponse):
    """Message class for OpenAI compatibility."""
    pass
