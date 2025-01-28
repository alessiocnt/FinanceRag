from typing import Dict, List
import torch
import numpy as np
from transformers import PreTrainedModel

class Embedder:
    def __init__(self, model: PreTrainedModel, device: str=None):
        """
        Initialize the Embedder with robust GPU optimization.
        :param model: Pretrained transformer model
        :param device: Explicitly set device (defaults to CUDA if available)
        """
        # Automatic device selection with preference for CUDA
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.model = model.to(self.device).eval() # Move model to selected device and set to evaluation mode

    def _generate_single_embedding(
        self, 
        chunk: Dict[str, torch.Tensor], 
        method: str = 'average_last_4'
    ) -> np.ndarray:
        """
        Generate embedding for a single preprocessed text chunk with error handling.
        :param chunk: Preprocessed text chunk with 'input_ids', 'attention_mask', 'token_type_ids'
        :param method: Embedding extraction method
        :return: Numpy array embedding
        """
        # Validate method
        if method not in ['last_layer', 'average_last_4']:
            raise ValueError("Invalid method. Choose between 'last_layer' and 'average_last_4'.")
        # Ensure all tensors are on the same device
        try:
            input_ids = chunk['input_ids'].to(self.device)
            attention_mask = chunk['attention_mask'].to(self.device)
            token_type_ids = chunk.get('token_type_ids')
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)
            else:
                token_type_ids = torch.zeros_like(input_ids, device=self.device)
            # Compute model outputs
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    output_hidden_states=True
                )
                # Embedding extraction
                if method == 'last_layer':
                    # CLS token from the last hidden layer
                    cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                elif method == 'average_last_4':
                    # Average of last 4 hidden layers
                    hidden_states = outputs.hidden_states
                    last_4_layers = torch.stack(hidden_states[-4:], dim=0)
                    avg_last_4 = torch.mean(last_4_layers, dim=0)
                    cls_embedding = avg_last_4[:, 0, :].cpu().numpy()
                return cls_embedding
        except Exception as e:
            print(f"Detailed error processing chunk: {e}")
            # Print additional debug information
            print(f"Original chunk keys: {chunk.keys()}")
            print(f"Input IDs original device: {chunk['input_ids'].device}")
            print(f"Attention mask original device: {chunk['attention_mask'].device}")
            if 'token_type_ids' in chunk:
                print(f"Token type IDs original device: {chunk['token_type_ids'].device}")
            raise

    def encode(
        self, 
        chunks: List[Dict[str, torch.Tensor]], 
        method: str = 'average_last_4', 
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Generate embeddings for preprocessed chunks with GPU batch processing.
        :param chunks: List of preprocessed text chunks
        :param method: Embedding extraction method
        :param batch_size: Number of chunks to process simultaneously
        :return: Numpy array of embeddings
        """
        # Validate input
        if not chunks:
            return np.array([])
        embeddings = []
        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            # Batch embedding generation with error handling
            batch_embeddings = []
            for chunk in batch_chunks:
                try:
                    embedding = self._generate_single_embedding(chunk, method)
                    batch_embeddings.append(embedding)
                except Exception as e:
                    print(f"Skipping chunk due to error: {e}")
            embeddings.extend(batch_embeddings)
        return np.vstack(embeddings) if embeddings else np.array([])

class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def tokenize(self, corpus: List[str]):
        tokens = []
        for item in corpus: 
            tokens.append(self.tokenizer(item, return_tensors="pt", padding=False))
        return tokens

# def chunker(tokenized_corpus: List[dict], max_length=512, padding_value=0):
#     """
#     Processes tokenized input and generates chunks while preserving the mapping structure.

#     Args:
#         tokenized_corpus (List[dict]): List of tokenized outputs from HuggingFace tokenizer.
#         max_length (int): Maximum chunk size.
#         padding_value (int): Value to pad incomplete chunks.

#     Returns:
#         List[dict]: List of chunked tokenized mappings ready for embedding.
#     """
#     chunks = []
#     for item in tokenized_corpus:
#         input_ids = item['input_ids'].squeeze(0)  # Remove batch dimension
#         attention_mask = item['attention_mask'].squeeze(0)  # Remove batch dimension
#         # Chunking
#         for i in range(0, input_ids.size(0), max_length):
#             input_chunk = input_ids[i:i + max_length]
#             attention_chunk = attention_mask[i:i + max_length]
#             # Padding
#             if input_chunk.size(0) < max_length:
#                 input_chunk = torch.nn.functional.pad(input_chunk, (0, max_length - input_chunk.size(0)), value=padding_value)
#                 attention_chunk = torch.nn.functional.pad(attention_chunk, (0, max_length - attention_chunk.size(0)), value=0)
#             # Append the chunk as a dictionary
#             chunks.append({
#                 'input_ids': input_chunk.unsqueeze(0),  # Add batch dimension
#                 'attention_mask': attention_chunk.unsqueeze(0)  # Add batch dimension
#             })
#     return chunks

from typing import List
import torch

def chunker(tokenized_corpus: List[dict], max_length=512, padding_value=0, overlap_percent=15):
    """
    Processes tokenized input and generates chunks with overlap while preserving the mapping structure.
    Args:
        tokenized_corpus (List[dict]): List of tokenized outputs from HuggingFace tokenizer.
        max_length (int): Maximum chunk size.
        padding_value (int): Value to pad incomplete chunks.
        overlap_percent (int): Percentage of overlap between chunks (1-100).
    Returns:
        List[dict]: List of chunked tokenized mappings ready for embedding.
    """
    if not 0 <= overlap_percent <= 100:
        raise ValueError("overlap_percent must be between 0 and 100")
    overlap_size = int(max_length * (overlap_percent / 100))
    stride = max_length - overlap_size
    chunks = []
    for item in tokenized_corpus:
        input_ids = item['input_ids'].squeeze(0)  # Remove batch dimension
        attention_mask = item['attention_mask'].squeeze(0)  # Remove batch dimension
        # Calculate total length and number of chunks needed
        total_length = input_ids.size(0)
        # Handle the case where the input is shorter than max_length
        if total_length <= max_length:
            input_chunk = input_ids
            attention_chunk = attention_mask
            # Padding
            if input_chunk.size(0) < max_length:
                input_chunk = torch.nn.functional.pad(
                    input_chunk, 
                    (0, max_length - input_chunk.size(0)), 
                    value=padding_value
                )
                attention_chunk = torch.nn.functional.pad(
                    attention_chunk, 
                    (0, max_length - attention_chunk.size(0)), 
                    value=0
                )
            chunks.append({
                'input_ids': input_chunk.unsqueeze(0),
                'attention_mask': attention_chunk.unsqueeze(0)
            })
            continue
        # Create overlapping chunks
        start_positions = range(0, total_length - overlap_size, stride)
        for start_pos in start_positions:
            end_pos = min(start_pos + max_length, total_length)
            # If this is the last chunk and it's too small, adjust start_pos
            if end_pos - start_pos < max_length and start_pos > 0:
                start_pos = max(0, end_pos - max_length)
            input_chunk = input_ids[start_pos:end_pos]
            attention_chunk = attention_mask[start_pos:end_pos]
            # Padding
            if input_chunk.size(0) < max_length:
                input_chunk = torch.nn.functional.pad(
                    input_chunk, 
                    (0, max_length - input_chunk.size(0)), 
                    value=padding_value
                )
                attention_chunk = torch.nn.functional.pad(
                    attention_chunk, 
                    (0, max_length - attention_chunk.size(0)), 
                    value=0
                )
            chunks.append({
                'input_ids': input_chunk.unsqueeze(0),
                'attention_mask': attention_chunk.unsqueeze(0)
            })
            # Break if we've reached the end of the sequence
            if end_pos >= total_length:
                break
    return chunks