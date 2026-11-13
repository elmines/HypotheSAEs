"""Utilities for computing text embeddings."""

import numpy as np
from typing import Dict, List, Optional, Sequence, Union
import concurrent.futures
from tqdm.auto import tqdm
import tiktoken
import os
import time
from pathlib import Path
import glob
import torch
import openai
from .utils import filter_invalid_texts
from sentence_transformers import SentenceTransformer
import gc
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Use environment variable for cache dir if set, otherwise use default
CACHE_DIR = os.getenv('EMB_CACHE_DIR') or os.path.join(Path(__file__).parent.parent, 'emb_cache')

def _embed_batch_openai(
        batch: List[str], 
        model: str, 
        client,
        max_tokens: int = 8192, 
        max_retries: int = 3, 
        backoff_factor: float = 3.0,
        timeout: Optional[float] = None
) -> List[List[float]]:
    """Helper function for batch embedding using OpenAI API."""
    # Truncate texts to max tokens
    enc = tiktoken.get_encoding("cl100k_base")  # encoding for OpenAI text-embedding models
    truncated_batch = []
    for text in batch:
        tokens = enc.encode(text.strip())
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            text = enc.decode(tokens)
        truncated_batch.append(text)
    
    for attempt in range(max_retries):
        try:
            request_kwargs = {
                "input": truncated_batch,
                "model": model,
            }
            if timeout is not None:
                request_kwargs["timeout"] = timeout
            response = client.embeddings.create(**request_kwargs)
            return [data.embedding for data in response.data]
            
        except (openai.RateLimitError, openai.APITimeoutError) as e:
            if attempt == max_retries - 1:  # Last attempt
                raise e
            
            base_wait = timeout if timeout is not None else 1.0
            wait_time = base_wait * (backoff_factor ** attempt)
            if attempt > 0:
                print(f"API error: {e}; retrying in {wait_time:.1f}s... ({attempt + 1}/{max_retries})")
            time.sleep(wait_time)


def load_embedding_cache(cache_name: str) -> dict:
    """Load cached embeddings from chunked files."""
    if not cache_name:
        return {}
    
    cache_dir = f"{CACHE_DIR}/{cache_name}"
    if not os.path.exists(cache_dir):
        return {}
    
    text2embedding = {}
    chunk_files = sorted(glob.glob(f"{cache_dir}/chunk_*.npy"))
    
    start_time = time.time()
    for chunk_file in tqdm(chunk_files, desc="Loading embedding chunks"):
        # Each chunk file contains a list of (text, embedding) tuples
        chunk_data = np.load(chunk_file, allow_pickle=True)
        for text, emb in chunk_data:
            text2embedding[text] = emb
            
    load_time = time.time() - start_time
    print(f"Loaded {len(text2embedding)} embeddings in {load_time:.1f}s")
            
    return text2embedding


def update_embedding_cache(cache_name: str, text2embedding: dict, chunk_size: int = 50000) -> None:
    """Update cache files in chunks."""
    if not cache_name:
        return
        
    cache_dir = f"{CACHE_DIR}/{cache_name}"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Convert to list of (text, embedding) tuples for storage
    items = list(text2embedding.items())
    
    for i in range(0, len(items), chunk_size):
        chunk_items = items[i:i + chunk_size]
        chunk_num = i // chunk_size
        chunk_path = f"{cache_dir}/chunk_{chunk_num:03d}.npy"
        np.save(chunk_path, np.array(chunk_items, dtype=object))
        

def _get_next_chunk_index(cache_name: str) -> int:
    """Determine the next available chunk index for a cache."""
    if not cache_name:
        return 0
        
    cache_dir = f"{CACHE_DIR}/{cache_name}"
    if not os.path.exists(cache_dir):
        return 0
        
    chunk_files = glob.glob(f"{cache_dir}/chunk_*.npy")
    if not chunk_files:
        return 0
        
    indices = [int(os.path.basename(f).split("_")[1].split(".")[0]) for f in chunk_files]
    return max(indices) + 1
    
def _save_embedding_chunk(cache_name: str, chunk_embeddings: dict, chunk_idx: int) -> int:
    """Save a chunk of embeddings to disk."""
    if not cache_name or not chunk_embeddings:
        return chunk_idx
        
    cache_dir = f"{CACHE_DIR}/{cache_name}"
    os.makedirs(cache_dir, exist_ok=True)
    
    chunk_path = f"{cache_dir}/chunk_{chunk_idx:03d}.npy"
    chunk_items = list(chunk_embeddings.items())
    np.save(chunk_path, np.array(chunk_items, dtype=object))
    print(f"Saved {len(chunk_items)} embeddings to {chunk_path}")
    
    return chunk_idx + 1

def get_openai_embeddings(
    texts: List[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 256,
    n_workers: int = 5,
    cache_name: Optional[str] = None,
    show_progress: bool = True,
    chunk_size: int = 50000,
    timeout: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Get embeddings using OpenAI API with parallel processing and chunked caching."""
    # Filter out None values and empty strings
    texts = filter_invalid_texts(texts)
    
    # Setup cache
    text2embedding = load_embedding_cache(cache_name)
    texts_to_embed = [text for text in texts if text not in text2embedding]
    
    if not texts_to_embed:
        return text2embedding
    
    from .llm_api import get_client
    client = get_client()
    
    # Process in chunks
    next_chunk_idx = _get_next_chunk_index(cache_name)
    
    # Create chunk ranges
    chunk_ranges = [(i, min(i+chunk_size, len(texts_to_embed))) 
                   for i in range(0, len(texts_to_embed), chunk_size)]
    
    # Outer progress bar for chunks
    chunk_iterator = chunk_ranges
    if show_progress:
        chunk_iterator = tqdm(chunk_iterator, desc="Processing chunks", total=len(chunk_ranges))
    
    for chunk_start, chunk_end in chunk_iterator:
        chunk_texts = texts_to_embed[chunk_start:chunk_end]
        chunk_embeddings = {}
        
        # Process chunk in batches with parallel workers
        batches = [chunk_texts[i:i+batch_size] for i in range(0, len(chunk_texts), batch_size)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for batch in batches:
                futures.append(executor.submit(_embed_batch_openai, batch, model, client, timeout=timeout))
            
            # Process results as they complete
            iterator = concurrent.futures.as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(batches), desc=f"Chunk {next_chunk_idx}")
                
            for future in iterator:
                batch_result = future.result()
                batch_idx = futures.index(future)
                batch = batches[batch_idx]
                
                for text, embedding in zip(batch, batch_result):
                    chunk_embeddings[text] = embedding
                    text2embedding[text] = embedding
        
        # Save completed chunk
        next_chunk_idx = _save_embedding_chunk(cache_name, chunk_embeddings, next_chunk_idx)
    
    return text2embedding

def get_local_embeddings(
    texts: List[str],
    model: str = "nomic-ai/modernbert-embed-base",
    batch_size: int = 128,
    show_progress: bool = True,
    cache_name: Optional[str] = None,
    chunk_size: int = 50000,
    device: Optional[torch.device] = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    return_stacked: bool = False,
) -> Union[Dict[str, np.ndarray], np.ndarray]:
    """Get embeddings using local SentenceTransformer model with chunked caching.

    By default, return a mapping from text to embedding. If ``return_stacked`` is
    true, return embeddings as a NumPy array in the order of ``texts`` instead.
    """
    # Filter out None values and empty strings
    texts = filter_invalid_texts(texts)
    
    # Setup cache
    text2embedding = load_embedding_cache(cache_name)
    texts_to_embed = [text for text in texts if text not in text2embedding]
    
    if not texts_to_embed:
        if return_stacked:
            return np.stack([text2embedding[text] for text in texts])
        return text2embedding

    # Enable TF32-backed float32 matmul on supported GPUs to avoid repeated warnings
    # from torch compile paths and improve embedding throughput.
    torch.set_float32_matmul_precision("high")

    # Load model
    transformer_model = SentenceTransformer(model, device=device)
    print(f"Loaded model {model} to {device}")
    
    # Process in chunks
    next_chunk_idx = _get_next_chunk_index(cache_name)
    
    # Create chunk ranges
    chunk_ranges = [(i, min(i+chunk_size, len(texts_to_embed))) 
                   for i in range(0, len(texts_to_embed), chunk_size)]
    
    # Outer progress bar for chunks
    chunk_iterator = chunk_ranges
    if show_progress:
        chunk_iterator = tqdm(chunk_iterator, desc="Processing chunks", total=len(chunk_ranges))
    
    for chunk_start, chunk_end in chunk_iterator:
        chunk_texts = texts_to_embed[chunk_start:chunk_end]
        chunk_embeddings = {}
        
        # Process chunk in batches
        batch_iterator = range(0, len(chunk_texts), batch_size)
        if show_progress:
            batch_iterator = tqdm(batch_iterator, desc=f"Chunk {next_chunk_idx}")
            
        for i in batch_iterator:
            batch = chunk_texts[i:i+batch_size]
            if "nomic-ai" in model:
                prefixed_batch = ["clustering: " + text for text in batch]
            elif "instructor" in model:
                prefixed_batch = [["Represent the text for classification: ", text] for text in batch]
            else:
                prefixed_batch = batch
            batch_embs = transformer_model.encode(prefixed_batch, batch_size=batch_size)
            
            for text, embedding in zip(batch, batch_embs):
                chunk_embeddings[text] = embedding
                text2embedding[text] = embedding
        
        # Save completed chunk
        next_chunk_idx = _save_embedding_chunk(cache_name, chunk_embeddings, next_chunk_idx)

    del transformer_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if return_stacked:
        return np.stack([text2embedding[text] for text in texts])
    return text2embedding


def get_joint_stance_embeddings(
    targets: Sequence[str],
    posts: Sequence[str],
    model: str = "NLP-Debater-Project/debertav3-stance-detection",
    batch_size: int = 8,
    max_length: int = 512,
    show_progress: bool = True,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Embed target/post pairs with a stance-detection model.

    Each target/post pair is formatted as ``Topic: {target} [SEP] Argument:
    {post}``, matching the model card's recommended input format. Embeddings
    are mean pooled from the model's final encoder layer and returned in input
    order.
    """
    if len(targets) != len(posts):
        raise ValueError("targets and posts must have the same length.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if max_length <= 0:
        raise ValueError("max_length must be positive.")
    if any(not isinstance(text, str) or not text.strip() for text in (*targets, *posts)):
        raise ValueError("targets and posts must contain non-empty strings.")

    embedding_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model)
    stance_model = AutoModelForSequenceClassification.from_pretrained(model)
    stance_model.to(embedding_device)
    stance_model.eval()

    embeddings = []
    batch_starts = range(0, len(targets), batch_size)
    if show_progress:
        batch_starts = tqdm(batch_starts, desc="Embedding target/post pairs")

    with torch.inference_mode():
        for start in batch_starts:
            batch_texts = [
                f"Topic: {target} [SEP] Argument: {post}"
                for target, post in zip(
                    targets[start:start + batch_size],
                    posts[start:start + batch_size],
                )
            ]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            inputs = {key: value.to(embedding_device) for key, value in inputs.items()}
            outputs = stance_model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            embeddings.append(pooled.cpu().numpy())

    hidden_size = int(stance_model.config.hidden_size)
    del stance_model
    gc.collect()
    if embedding_device.type == "cuda":
        torch.cuda.empty_cache()

    if not embeddings:
        return np.empty((0, hidden_size), dtype=np.float32)
    return np.concatenate(embeddings, axis=0).astype(np.float32, copy=False)
