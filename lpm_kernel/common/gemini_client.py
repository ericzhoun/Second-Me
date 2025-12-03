import google.generativeai as genai
import logging
import time
import uuid
from typing import List, Dict, Any, Optional, Union, Iterator

logger = logging.getLogger(__name__)

class GeminiClient:
    """
    Adapter for Google Gemini API to mimic OpenAI client interface.
    """
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        if not api_key:
            raise ValueError("API key is required for GeminiClient")
        genai.configure(api_key=api_key)
        self.base_url = base_url or "https://generativelanguage.googleapis.com"
        self.chat = self.Chat(self)

    class Chat:
        def __init__(self, client):
            self.client = client
            self.completions = self.Completions(client)

        class Completions:
            def __init__(self, client):
                self.client = client

            def create(self, **kwargs) -> Any:
                return self.client._create_completion(**kwargs)

    def _convert_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Convert OpenAI messages to Gemini history format."""
        gemini_history = []
        system_instruction = None

        # Extract system prompt if present (Gemini supports system_instruction at model init)
        # However, generate_content doesn't support system_instruction per call easily unless we use beta or specific models.
        # Standard approach: Merge system prompt into first user message or use system_instruction if model supports it.
        # For simplicity and broad support, we can prepend system prompt.
        # UPDATE: Gemini 1.5 Pro/Flash supports system_instruction.

        # Let's separate system messages
        system_messages = [m for m in messages if m.get("role") == "system"]
        if system_messages:
            system_instruction = "\n".join([m.get("content", "") for m in system_messages])

        # Process user/assistant messages
        # Gemini expects 'user' role as 'user' and 'assistant' as 'model'.
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "system":
                continue # Handled separately

            if role == "user":
                gemini_history.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                gemini_history.append({"role": "model", "parts": [content]})

        return gemini_history, system_instruction

    def _create_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Any:
        try:
            # Handle model name (remove 'models/' prefix if present twice or ensure it matches Gemini format)
            # Gemini models are usually "models/gemini-1.5-flash" or just "gemini-1.5-flash"
            # If user provides "models/lpm", we might need to fallback or trust config.
            # Assuming config provides valid Gemini model name.

            # Convert messages
            history, system_instruction = self._convert_messages(messages)

            # Configure model
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )

            gemini_model = genai.GenerativeModel(
                model_name=model,
                system_instruction=system_instruction,
                generation_config=generation_config
            )

            # Prepare chat or content generation
            # If history is empty (only system prompt?), sending empty content might fail.
            # If history has only one user message, use generate_content.
            # If history has multiple, use start_chat.

            if not history:
                # Should not happen in valid chat
                raise ValueError("No user/model messages provided")

            last_message = history[-1]
            if last_message["role"] != "user":
                # OpenAI allows last message to be assistant (to continue?), Gemini expects user prompt last?
                # Actually generate_content takes content.
                # If we use chat, we need history + current message.
                pass

            # We will use start_chat for history support
            # Pop the last message as the new prompt
            if history and history[-1]["role"] == "user":
                prompt = history[-1]["parts"][0]
                chat_history = history[:-1]
            else:
                # Fallback if last message is not user (e.g. continue generation? Not fully supported here)
                prompt = " " # Empty prompt?
                chat_history = history

            chat_session = gemini_model.start_chat(history=chat_history)

            response = chat_session.send_message(prompt, stream=stream)

            if stream:
                return self._stream_response_adapter(response, model)
            else:
                return self._response_adapter(response, model)

        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            raise

    def _response_adapter(self, response, model):
        """Adapt Gemini response to OpenAI format object."""
        # Wait for response completion
        try:
            text = response.text
        except ValueError:
            # Blocked content?
            text = ""
            if response.prompt_feedback:
                logger.warning(f"Gemini prompt feedback: {response.prompt_feedback}")

        return OpenAIResponse({
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": OpenAIMessage({
                    "role": "assistant",
                    "content": text
                }),
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0, # Not easily available
                "completion_tokens": 0,
                "total_tokens": 0
            }
        })

    def _stream_response_adapter(self, response_iterator, model):
        """Yield OpenAI-format chunks from Gemini stream."""
        response_id = f"chatcmpl-{uuid.uuid4()}"
        created = int(time.time())

        for chunk in response_iterator:
            text = ""
            try:
                text = chunk.text
            except ValueError:
                continue

            yield OpenAIResponse({
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": text
                    },
                    "finish_reason": None
                }]
            })

        # Yield finish reason
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

class OpenAIResponse(dict):
    """Helper to allow dot access to dictionary."""
    def __init__(self, data):
        super().__init__(data)
        for k, v in data.items():
            if isinstance(v, dict):
                self[k] = OpenAIResponse(v)
            elif isinstance(v, list):
                self[k] = [OpenAIResponse(i) if isinstance(i, dict) else i for i in v]

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

class OpenAIMessage(OpenAIResponse):
    pass
