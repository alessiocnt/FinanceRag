import torch
import numpy as np

class EmbeddingGenerator:
    def __init__(self, tokenizer, model, method=2, chunck_size=16):
        self.tokenizer = tokenizer
        self.model = model
        self.method = method
        self.chunck_size = chunck_size

    def generate_single_embedding(self, chunk):
        self.model.eval()
        with torch.no_grad():
            # STEP 1: Tokenize the text chunck
            inputs = self.tokenizer(chunk, padding=True, truncation=True, return_tensors="pt", max_length=512)
            
            # STEP 2: forward pass through the model
            outputs = self.model(**inputs, output_hidden_states=True)

            # STEP 3: Extract the embeddings (choose between 2 methods)
            # Method 1: select the CLS of the last hidden layer 
            if self.method == 1:  
                cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()    # (chunck_size, seq_length, hidden_size)
            # Method 2: Ottieni gli ultimi 4 hidden states
            if self.method == 2:
                hidden_states = outputs.hidden_states  
                last_4_layers = hidden_states[-4:]  
                # Calcola la media dei 4 layer
                stacked_layers = torch.stack(last_4_layers, dim=0) 
                avg_last_4 = torch.mean(stacked_layers, dim=0)      
                # Select token [CLS] of the average
                cls_embedding = avg_last_4[:, 0, :].squeeze().numpy()  
        return cls_embedding

    def generate_embeddings(self, text):
        self.model.eval()
        embedding_list = []
        # no gradient calculation
        with torch.no_grad():
            for i in range(0, len(text), self.chunck_size):
                chunck_texts = text[i:i + self.chunck_size]
                cls_embedding = self.generate_single_embedding(chunck_texts)
                # Append the embedding to the list
                embedding_list.append(cls_embedding)
        return np.vstack(embedding_list)

