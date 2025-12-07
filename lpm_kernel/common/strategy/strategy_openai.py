from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import numpy as np
from lpm_kernel.configs.logging import get_train_process_logger
import requests

if TYPE_CHECKING:
    from lpm_kernel.api.dto.user_llm_config_dto import UserLLMConfigDTO

logger = get_train_process_logger()

def openai_strategy(user_llm_config: Optional["UserLLMConfigDTO"], chunked_texts):
    try:
        headers = {
            "Authorization": f"Bearer {user_llm_config.embedding_api_key}",
            "Content-Type": "application/json",
        }

        data = {"input": chunked_texts, "model": user_llm_config.embedding_model_name}

        logger.info(f"Getting embedding for {data}, total chunks: {len(chunked_texts)}")

        response = requests.post(
            f"{user_llm_config.embedding_endpoint}/embeddings", headers=headers, json=data
        )
        response.raise_for_status()
        result = response.json()

        # Extract embedding vectors
        embeddings = [item["embedding"] for item in result["data"]]
        embeddings_array = np.array(embeddings)

        return embeddings_array

    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to get embeddings: {str(e)}") from e

