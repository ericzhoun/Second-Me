from itertools import islice
import concurrent.futures
import json
import os
import random
import re
from tqdm import tqdm
import openai
from enum import Enum
from lpm_kernel.api.services.user_llm_config_service import UserLLMConfigService
from lpm_kernel.configs.config import Config
from lpm_kernel.L2.data_pipeline.data_prep.preference.prompts import (
    CH_USR_TEMPLATES, CH_USR_COT_TEMPLATES,
    EN_USR_TEMPLATES, EN_USR_COT_TEMPLATES,
    CH_SYS_TEMPLATES, CH_SYS_COT_TEMPLATES,
    EN_SYS_TEMPLATES, EN_SYS_COT_TEMPLATES
)
import traceback
from lpm_kernel.common.gemini_client import GeminiClient
from lpm_kernel.configs.logging import get_train_process_logger
logger = get_train_process_logger()

class TqdmLoggingHandler:
    def __init__(self):
        pass
    
    def write(self, msg):
        logger.info(msg.strip())
    
    def flush(self):
        pass
    
tqdm_handler = TqdmLoggingHandler()


class LowMode(Enum):
    cluster_nums = 3


class MediumMode(Enum):
    cluster_nums = 2


class HighMode(Enum):
    cluster_nums = 1


class PreferenceQAGenerator:
    def __init__(self, filename: str, bio: str, preference_language: str, is_cot: bool = True):
        """Initialize the PreferenceQAGenerator class.
        
        Args:
            filename: Path to the input JSON file containing preference messages.
            bio: Biography or context information to use in prompt generation.
            preference_language: Language for prompts ("Chinese/中文" or otherwise English).
        """
        # Ensure the filename is actually a string
        if filename is None:
            raise ValueError("Filename cannot be None")
            
        self.filename = filename
        # Convert is_cot to bool if it's a string
        if isinstance(is_cot, str):
            self.is_cot = is_cot.lower() == 'true'
        else:
            self.is_cot = bool(is_cot)
            
        logger.info(f"PreferenceQAGenerator initialized with is_cot={self.is_cot}")
        
        with open(self.filename, "r", encoding="utf-8") as f:
            self.pre_msg = json.load(f)

        user_llm_config_service = UserLLMConfigService()
        user_llm_config = user_llm_config_service.get_available_llm()
        if user_llm_config is None:
            self.client = None
            self.model_name = None
        else:
            self.model_name = user_llm_config.chat_model_name

            if user_llm_config.provider_type == 'gemini':
                logger.info("Initializing Gemini client for PreferenceQA generation")
                self.client = GeminiClient(
                    api_key=user_llm_config.chat_api_key,
                    base_url=user_llm_config.chat_endpoint
                )
            else:
                self.client = openai.OpenAI(
                    api_key=user_llm_config.chat_api_key,
                    base_url=user_llm_config.chat_endpoint,
                )

        if self.is_cot:
            logger.info("generate pereference data in longcot pattern!!!")
            self.model_name = user_llm_config.thinking_model_name
            self.api_key = user_llm_config.thinking_api_key
            self.base_url = user_llm_config.thinking_endpoint
            if self.model_name.startswith("deepseek"):
                self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
            elif user_llm_config.provider_type == 'gemini':
                # For Gemini, assume thinking models are used if model name indicates it or just reuse main logic
                # Gemini doesn't strictly need a separate client if api key is same, but to be consistent with logic:
                # If thinking model is specified and provider is Gemini, reuse GeminiClient with thinking credentials?
                # Actually if provider is gemini, user might not have set 'thinking_model_name' to deepseek.
                # If they did, then it's a conflict.
                # Assuming if is_cot is true and provider is gemini, we use the Gemini client with thinking model name if provided.
                if self.model_name:
                    logger.info(f"Using thinking model for Gemini: {self.model_name}")
                    # Client already initialized with Gemini
                    pass
            else:
                logger.error(f"Error model_name, longcot data generating model_name: deepseek series")
                raise
            
        
        self.bio = bio
        self.question_list = []
        self.preference_language = preference_language
        self.prompt_templates = self._get_prompt_templates(preference_language)
        self.sys_templates = self._get_sys_templates(preference_language)
        self.max_workers = 1
        self.data_synthesis_mode = os.environ.get("DATA_SYNTHESIS_MODE", "low")


    def generate_response(self, sys: str, prompt: str) -> str:
        """Generate a response using the OpenAI / DeepSeek API.
        
        Args:
            sys: The system prompt to use.
            prompt: The user prompt to send to the API.
            
        Returns:
            The generated response text or None if an error occurred.
        """
        def get_remote_response(sys: str, prompt: str) -> str:
            """Get response from OpenAI / DeepSeek API.
            
            Args:
                sys: The system prompt to use.
                prompt: The user prompt to send to the API.
                
            Returns:
                The response content from OpenAI / DeepSeek, or None if an error occurs.
            """
            try:
                res = self.client.chat.completions.create(
                    messages=[
                            {"role": "system", "content": sys},
                            {"role": "user", "content": prompt},
                        ],
                    model=self.model_name,
                )
                response_message = res.choices[0].message
                if self.is_cot:
                    return "<think>" + response_message.reasoning_content + "</think>" + response_message.content
                else:
                    return response_message.content
            except Exception as e:
                logger.error(traceback.format_exc())
            return None
        
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future = executor.submit(
                    self.client.chat.completions.create,
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": sys},
                        {"role": "user", "content": prompt},
                    ],
                )
                response = future.result()
                return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return None


    def clean_chunk(self, chunk: str) -> str:
        """Clean and process a text chunk.
        
        Args:
            chunk: The text chunk to clean.
            
        Returns:
            Cleaned text after processing.
        """
        after = chunk.split(":")[1:]
        after = ":".join(after)
        return after[1:]


    def _get_prompt_templates(self, preference_language: str) -> dict:
        """Return a dictionary of prompt templates based on language preference (w or w/o cot).
        
        Args:
            preference_language: The language preference ("Chinese/中文" or otherwise English).
            
        Returns:
            A dictionary of prompt templates.
        """
        if preference_language == "Chinese":
            if self.is_cot:
                return CH_USR_COT_TEMPLATES
            else:
                return CH_USR_TEMPLATES
        else:
            if self.is_cot:
                return EN_USR_COT_TEMPLATES
            else:
                return EN_USR_TEMPLATES


    def _get_sys_templates(self, preference_language: str) -> dict:
        """Return a dictionary of system templates based on language preference (w or w/o cot).
        
        Args:
            preference_language: The language preference ("Chinese/中文" or otherwise English).
            
        Returns:
            A dictionary of system templates.
        """
        if preference_language == "Chinese":
            if self.is_cot:
                return CH_SYS_COT_TEMPLATES
            else:
                return CH_SYS_TEMPLATES
        else:
            if self.is_cot:
                return EN_SYS_COT_TEMPLATES
            else:
                return EN_SYS_TEMPLATES


    def process_clusters(self, output_filename: str) -> None:
        """Process clusters and generate questions and answers.
        
        Args:
            output_filename: Path to save the generated Q&A pairs.
        """
        cluster_items = list(self.pre_msg.items())
        count = 0
        
        if self.data_synthesis_mode == "low":
            sample_num = max(1, len(cluster_items) // LowMode.cluster_nums.value) if 0 < len(cluster_items) < 3 else len(cluster_items) // LowMode.cluster_nums.value
            new_cluster_items = random.sample(cluster_items, sample_num)
        elif self.data_synthesis_mode == "medium":
            sample_num = max(1, len(cluster_items) // MediumMode.cluster_nums.value) if 0 < len(cluster_items) < 2 else len(cluster_items) // MediumMode.cluster_nums.value
            new_cluster_items = random.sample(cluster_items, sample_num)
        else: # high or other case
            new_cluster_items = cluster_items
            
        for _, cluster in tqdm(new_cluster_items, desc="preference_generate", file=tqdm_handler):
            chunk_concat = self._get_chunk_concat(cluster["contents"])

            tags = " ".join(cluster["tags"])

            if len(chunk_concat) < 20:
                continue
            count += 1
            
            n_cluster = len(cluster["contents"])
            if n_cluster > 1:
                logger.info(f"Cluster has {str(n_cluster)} chunks")

            prompt_question_template = self.prompt_templates["query"]
            prompt_answer_template = self.prompt_templates["answer"]
            sys_question = self.sys_templates["query"]
            sys_answer = self.sys_templates["answer"]

            try:
                gen_question = self.generate_response(
                    sys_question,
                    prompt_question_template.format(
                        bio=self.bio, chunks_concat=chunk_concat
                    ),
                )
                if self.is_cot:
                    question_match = re.search(r"<question>(.*?)</question>", gen_question, re.DOTALL)
                    gen_question = question_match.group(1).strip() if question_match else gen_question
            except Exception as e:
                logger.error(traceback.format_exc())
                continue
            try:
                gen_answer = self.generate_response(
                    sys_answer,
                    prompt_answer_template.format(
                        question=gen_question, bio=self.bio, chunks_concat=chunk_concat
                    ),
                )
            except Exception as e:
                logger.error(traceback.format_exc())
                continue
            
            self.question_list.append({"user": gen_question, "assistant": gen_answer})
            if n_cluster >= 20:
                self._generate_multiple_questions(cluster["contents"], chunk_concat)
            if count % 5 == 0:
                logger.info(f"Processed {count} clusters")

        with open(output_filename, "w") as json_file:
            json.dump(self.question_list, json_file, indent=4, ensure_ascii=False)


    def _get_chunk_concat(self, contents: list) -> str:
        """Concatenate and clean chunks of text.
        
        Args:
            contents: List of content chunks to concatenate.
            
        Returns:
            Concatenated text with formatting.
        """
        chunk_concat = ""
        for content in contents:
            chunk_content = content
            chunk_concat += chunk_content
            chunk_concat += "\n\n"
        return chunk_concat


    def _generate_multiple_questions(self, contents: list, chunk_concat: str) -> None:
        """Generate multiple questions and answers for larger clusters.
        
        Args:
            contents: List of content chunks.
            chunk_concat: Concatenated text chunks.
        """
        num_chunk_referred = 30
        n_repeat = max(1, int(len(contents) * 1 / num_chunk_referred))
        chunk_content_list = [
            self.clean_chunk(content)
            for content in contents
            if len(self.clean_chunk(content)) >= 80
        ]

        logger.info(f"Big cluster: n_repeat = {n_repeat}")

        for i in range(n_repeat):
            if i % 5 == 0 and i > 0:
                logger.info(f"Repeat {i} times")
            selected_chunks = random.sample(
                chunk_content_list, min(len(chunk_content_list), num_chunk_referred)
            )
            chunk_concat = "\n".join(selected_chunks)
            prompt_question_template = self.prompt_templates["query"]
            prompt_answer_template = self.prompt_templates["answer"]
            sys_question = self.sys_templates["query"]
            sys_answer = self.sys_templates["answer"]

            try:
                gen_question = self.generate_response(
                    sys_question,
                    prompt_question_template.format(
                        bio=self.bio, chunks_concat=chunk_concat
                    ),
                )
                if self.is_cot:
                    question_match = re.search(r"<question>(.*?)</question>", gen_question, re.DOTALL)
                    gen_question = question_match.group(1).strip() if question_match else gen_question
                gen_answer = self.generate_response(
                    sys_answer,
                    prompt_answer_template.format(
                        question=gen_question, chunks_concat=chunk_concat, bio=self.bio
                    ),
                )
            except Exception as e:
                logger.error(traceback.format_exc())
                continue
            self.question_list.append({"user": gen_question, "assistant": gen_answer})
        return


    def process_clusters_batch(self, output_filename: str, batch_size: int = 10) -> None:
        """Process clusters in batches to reduce API calls.
        
        Instead of making 2 API calls per cluster (question + answer), this method
        batches multiple clusters into a single request, asking the LLM to generate
        Q&A pairs for all clusters at once.
        
        Args:
            output_filename: Path to save the generated Q&A pairs.
            batch_size: Number of clusters to process per API call. Default 10.
                        Note: Gemini may return empty responses for large batches.
                        Recommended: 5-15 for Gemini, 20-50 for other models.
        """
        from lpm_kernel.L2.data_pipeline.data_prep.preference.prompts import (
            EN_BATCH_SYS_TEMPLATE, EN_BATCH_USR_TEMPLATE,
            CH_BATCH_SYS_TEMPLATE, CH_BATCH_USR_TEMPLATE
        )
        
        cluster_items = list(self.pre_msg.items())
        
        # === RESUME LOGIC: Load existing progress ===
        processed_cluster_ids = set()
        progress_file = output_filename + ".progress"
        
        # Load existing Q&A pairs if output file exists
        if os.path.exists(output_filename):
            try:
                with open(output_filename, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                    if isinstance(existing_data, list):
                        self.question_list = existing_data
                        logger.info(f"Loaded {len(existing_data)} existing Q&A pairs from {output_filename}")
            except Exception as e:
                logger.warning(f"Could not load existing output file: {e}")
        
        # Load processed cluster IDs from progress file
        if os.path.exists(progress_file):
            try:
                with open(progress_file, "r", encoding="utf-8") as f:
                    processed_cluster_ids = set(json.load(f))
                    logger.info(f"Resuming: {len(processed_cluster_ids)} clusters already processed")
            except Exception as e:
                logger.warning(f"Could not load progress file: {e}")
        
        # Apply sampling based on data synthesis mode
        if self.data_synthesis_mode == "low":
            sample_num = max(1, len(cluster_items) // LowMode.cluster_nums.value) if 0 < len(cluster_items) < 3 else len(cluster_items) // LowMode.cluster_nums.value
            cluster_items = random.sample(cluster_items, sample_num)
        elif self.data_synthesis_mode == "medium":
            sample_num = max(1, len(cluster_items) // MediumMode.cluster_nums.value) if 0 < len(cluster_items) < 2 else len(cluster_items) // MediumMode.cluster_nums.value
            cluster_items = random.sample(cluster_items, sample_num)
        
        # Filter out already-processed clusters
        remaining_clusters = [(cid, cluster) for cid, cluster in cluster_items if cid not in processed_cluster_ids]
        skipped_count = len(cluster_items) - len(remaining_clusters)
        
        if skipped_count > 0:
            logger.info(f"Skipping {skipped_count} already-processed clusters, {len(remaining_clusters)} remaining")
        
        if not remaining_clusters:
            logger.info("All clusters already processed. Nothing to do.")
            return
        
        # Select templates based on language
        if self.preference_language == "Chinese":
            batch_sys = CH_BATCH_SYS_TEMPLATE
            batch_usr = CH_BATCH_USR_TEMPLATE
        else:
            batch_sys = EN_BATCH_SYS_TEMPLATE
            batch_usr = EN_BATCH_USR_TEMPLATE
        
        total_clusters = len(remaining_clusters)
        num_batches = (total_clusters + batch_size - 1) // batch_size
        
        logger.info(f"Processing {total_clusters} clusters in {num_batches} batches (batch_size={batch_size})")
        
        processed_count = 0
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_clusters)
            batch = remaining_clusters[start_idx:end_idx]
            
            # Track cluster IDs in this batch for progress saving
            batch_cluster_ids = [cid for cid, _ in batch]
            
            # Build clusters content for prompt
            clusters_content = ""
            valid_clusters = []
            
            for i, (cluster_id, cluster) in enumerate(batch):
                chunk_concat = self._get_chunk_concat(cluster["contents"])
                if len(chunk_concat) < 20:
                    continue
                clusters_content += f"\n### Cluster {i} ###\n{chunk_concat}\n"
                valid_clusters.append(i)
            
            if not valid_clusters:
                logger.info(f"Batch {batch_idx + 1}/{num_batches}: No valid clusters, skipping")
                continue
            
            # Build prompt
            prompt = batch_usr.format(
                bio=self.bio,
                num_clusters=len(valid_clusters),
                clusters_content=clusters_content
            )
            
            logger.info(f"Batch {batch_idx + 1}/{num_batches}: Processing {len(valid_clusters)} clusters...")
            
            try:
                response = self.generate_response(batch_sys, prompt)
                
                if response is None or response.strip() == "":
                    logger.warning(f"Batch {batch_idx + 1}/{num_batches}: Empty response, falling back to individual processing")
                    # Fallback: process this batch's clusters individually
                    for cluster_id, cluster in batch:
                        chunk_concat = self._get_chunk_concat(cluster["contents"])
                        if len(chunk_concat) < 20:
                            continue
                        try:
                            # Use original single-cluster processing
                            gen_question = self.generate_response(
                                self.sys_templates["query"],
                                self.prompt_templates["query"].format(bio=self.bio, chunks_concat=chunk_concat)
                            )
                            if gen_question:
                                gen_answer = self.generate_response(
                                    self.sys_templates["answer"],
                                    self.prompt_templates["answer"].format(
                                        question=gen_question, bio=self.bio, chunks_concat=chunk_concat
                                    )
                                )
                                if gen_answer:
                                    self.question_list.append({"user": gen_question, "assistant": gen_answer})
                                    processed_count += 1
                        except Exception as e:
                            logger.error(f"Individual fallback failed for cluster: {e}")
                            continue
                    continue
                
                # Parse JSON response
                qa_pairs = self._parse_batch_response(response)
                
                if qa_pairs:
                    for qa in qa_pairs:
                        if "question" in qa and "answer" in qa:
                            self.question_list.append({
                                "user": qa["question"],
                                "assistant": qa["answer"]
                            })
                            processed_count += 1
                    
                    logger.info(f"Batch {batch_idx + 1}/{num_batches}: Generated {len(qa_pairs)} Q&A pairs")
                else:
                    logger.warning(f"Batch {batch_idx + 1}/{num_batches}: Failed to parse response, trying smaller batch")
                    
            except Exception as e:
                logger.error(f"Batch {batch_idx + 1}/{num_batches}: Error - {str(e)}")
                logger.error(traceback.format_exc())
                continue
            
            # === SAVE PROGRESS AFTER EACH BATCH ===
            # Mark these clusters as processed
            processed_cluster_ids.update(batch_cluster_ids)
            
            # Save Q&A pairs
            with open(output_filename, "w", encoding="utf-8") as json_file:
                json.dump(self.question_list, json_file, indent=4, ensure_ascii=False)
            
            # Save progress (processed cluster IDs)
            with open(progress_file, "w", encoding="utf-8") as f:
                json.dump(list(processed_cluster_ids), f)
            
            logger.info(f"Progress saved: {len(processed_cluster_ids)} clusters processed, {len(self.question_list)} Q&A pairs total")
        
        logger.info(f"Batch processing complete. Total Q&A pairs generated: {processed_count}")
        
        # Final save
        with open(output_filename, "w", encoding="utf-8") as json_file:
            json.dump(self.question_list, json_file, indent=4, ensure_ascii=False)


    def _parse_batch_response(self, response: str) -> list:
        """Parse the batch response to extract Q&A pairs.
        
        Args:
            response: The raw response from the LLM.
            
        Returns:
            List of dictionaries with 'question' and 'answer' keys.
        """
        try:
            # Try to find JSON array in response
            # First, try direct JSON parse
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                pass
            
            # Try to extract JSON from markdown code blocks
            json_match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', response)
            if json_match:
                return json.loads(json_match.group(1))
            
            # Try to find array pattern
            array_match = re.search(r'\[[\s\S]*\]', response)
            if array_match:
                return json.loads(array_match.group(0))
            
            logger.warning(f"Could not extract JSON from response: {response[:200]}...")
            return []
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.error(f"Response snippet: {response[:500]}...")
            return []
        except Exception as e:
            logger.error(f"Error parsing batch response: {e}")
            return []
