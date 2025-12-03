from lpm_kernel.api.dto.user_llm_config_dto import UserLLMConfigDTO
from lpm_kernel.configs.logging import get_train_process_logger
import google.generativeai as genai
import os
import numpy as np
from typing import List, Union

logger = get_train_process_logger()

def gemini_strategy(user_llm_config: UserLLMConfigDTO, chunked_texts: List[str]) -> np.ndarray:
    """
    Generate embeddings using Google Gemini API

    Args:
        user_llm_config: User LLM Configuration
        chunked_texts: List of text chunks to embed

    Returns:
        numpy.ndarray: Array of embeddings
    """
    try:
        # Get API key from env var (preferred) or config
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            # Fallback to config if available, though requirements emphasize env var
            api_key = user_llm_config.embedding_api_key or user_llm_config.key

        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        # Configure Gemini
        genai.configure(api_key=api_key)

        # Get model name, default to text-embedding-004 if not specified
        model_name = user_llm_config.embedding_model_name
        if not model_name:
            model_name = "models/text-embedding-004"
        elif not model_name.startswith("models/"):
             model_name = f"models/{model_name}"

        embeddings_list = []

        # Iterate over chunks to get embedding for each
        for text in chunked_texts:
            try:
                result = genai.embed_content(
                    model=model_name,
                    content=text,
                    task_type="retrieval_document",
                    title=None,
                    output_dimensionality=512 # Set default dimension to 512 as requested
                )

                if 'embedding' in result:
                    embeddings_list.append(result['embedding'])
                else:
                    logger.warning(f"No embedding found for chunk: {text[:50]}...")
                    # Fallback or error? For now, if one fails, maybe we should raise
                    # But keeping consistent size is important.
                    # If empty, maybe add a zero vector? Or raise.
                    raise ValueError(f"Unexpected response from Gemini API: {result}")
            except Exception as e:
                logger.error(f"Error embedding chunk '{text[:50]}...': {str(e)}")
                raise

        return np.array(embeddings_list)

    except Exception as e:
        logger.error(f"Error generating Gemini embeddings: {str(e)}")
        raise
