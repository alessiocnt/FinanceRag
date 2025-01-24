from typing import List
import torch
import numpy as np

class Embedder:
    def __init__(self, model, chunck_size=512):
        self.model = model
        self.chunck_size = chunck_size

    def _generate_single_embedding(self, chunk, method='average_last_4'):
        """Generate the embedding for a single text chunck.
        :param chunk: Text chunck to be processed.
        :return: Embedding of the text chunck.
        """
        if method not in ['last_layer', 'average_last_4']:
            raise ValueError("Invalid method. Choose between 'last_layer' and 'average_last_4'.")
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**chunk, output_hidden_states=True)
            # Method 1: select the CLS of the last hidden layer 
            if method == 'last_layer':  
                cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()    # (chunck_size, seq_length, hidden_size)
            # Method 2: Ottieni gli ultimi 4 hidden states
            if method == 'average_last_4':
                hidden_states = outputs.hidden_states  
                last_4_layers = hidden_states[-4:]  
                # Calcola la media dei 4 layer
                stacked_layers = torch.stack(last_4_layers, dim=0) 
                avg_last_4 = torch.mean(stacked_layers, dim=0)      
                # Select token [CLS] of the average
                cls_embedding = avg_last_4[:, 0, :].squeeze().numpy()  
        return cls_embedding

    def generate_embeddings(self, chunks: List[np.array]):
        self.model.eval()
        embeddings = []
        # no gradient calculation
        with torch.no_grad():
            for chunk in chunks:
                embeddings.append(self._generate_single_embedding(chunk))
        return np.vstack(embeddings)

class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def tokenize(self, corpus: List[str]):
        tokens = []
        for item in corpus: 
            tokens.append(self.tokenizer(item, return_tensors="pt", padding=False))
        return tokens

def chunker(tokenized_corpus: List[dict], max_length=512, padding_value=0):
    """
    Processes tokenized input and generates chunks while preserving the mapping structure.

    Args:
        tokenized_corpus (List[dict]): List of tokenized outputs from HuggingFace tokenizer.
        max_length (int): Maximum chunk size.
        padding_value (int): Value to pad incomplete chunks.

    Returns:
        List[dict]: List of chunked tokenized mappings ready for embedding.
    """
    chunks = []
    for item in tokenized_corpus:
        input_ids = item['input_ids'].squeeze(0)  # Remove batch dimension
        attention_mask = item['attention_mask'].squeeze(0)  # Remove batch dimension
        # Chunking
        for i in range(0, input_ids.size(0), max_length):
            input_chunk = input_ids[i:i + max_length]
            attention_chunk = attention_mask[i:i + max_length]
            # Padding
            if input_chunk.size(0) < max_length:
                input_chunk = torch.nn.functional.pad(input_chunk, (0, max_length - input_chunk.size(0)), value=padding_value)
                attention_chunk = torch.nn.functional.pad(attention_chunk, (0, max_length - attention_chunk.size(0)), value=0)
            # Append the chunk as a dictionary
            chunks.append({
                'input_ids': input_chunk.unsqueeze(0),  # Add batch dimension
                'attention_mask': attention_chunk.unsqueeze(0)  # Add batch dimension
            })
    return chunks

# def chunker(tokenized_corpus: List[torch.Tensor], max_length=512, padding_value=0):
#     chunks = []
#     for item in tokenized_corpus:
#         item = item.squeeze(0)  # Remove extra dimensions if needed (e.g., shape [1, N] -> [N])
#         for i in range(0, item.size(0), max_length):
#             chunk = item[i:i + max_length]
#             # Pad chunk if it's shorter than max_length
#             if chunk.size(0) < max_length:
#                 chunk = torch.nn.functional.pad(chunk, (0, max_length - chunk.size(0)), value=padding_value)
#             chunks.append(chunk)
#     return np.vstack(chunks)
