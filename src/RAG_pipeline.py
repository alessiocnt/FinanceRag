import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score
from data_handler import DataHandler
from dataset import *
from vector_store import FaissVectorStore

class RAGPipeline:
    def __init__(self, 
                 corpus: List[Dict], 
                 queries: List[Dict], 
                 qrels: pd.DataFrame,
                 text_processor: DataHandler):
        self.corpus = corpus
        self.queries = queries
        self.qrels = qrels
        self.text_processor = text_processor
        self.corpus_embedding = {}
        self.query_embedding = {}
        self.vector_store = None

    def load_table_summaries(self, load_path: str=None):
        table_summaries = np.load(load_path, allow_pickle='TRUE').item()
        # Add table summaries to corpus
        for idx, text in table_summaries.items():
            self.corpus[idx].append(text)
    
    def embed_corpus(self, chunk_size: int=256, method: str=None):
        for idx, text in self.corpus.items():
            embedded_doc = (
                self.text_processor.load_data(text)
                # .to_lowercase()
                # .remove_numbers()
                # .remove_punctuation()
                # .remove_stopwords()
                # .lemmatize_text()
                # .remove_extra_whitespace()
                .tokenize()
                .chunk_split(max_length=chunk_size)
                .embed(method=method)
                .get_data()
            )
            self.corpus_embedding[idx] = embedded_doc

    def populate_vector_store(self, 
                              embedding_dim: int=768,
                              save_path: str=None):
        self.vector_store = FaissVectorStore(embedding_dim)
        # Populate vector store
        for i, doc in self.corpus_embedding.items():
            self.vector_store.add_embeddings(np.array(doc), [i for e in range(len(doc))], [{'chunk': e} for e in range(len(doc))])
        # Save vector store to file
        if save_path:
            self.vector_store.save(save_path)

    def embed_queries(self, chunk_size: int=256, method: str=None):
        for idx, text in self.queries.items():
            embedded_query = (
                self.text_processor.load_data(text)
                # .to_lowercase()
                # .remove_numbers()
                # .remove_punctuation()
                # .remove_stopwords()
                # .lemmatize_text()
                # .remove_extra_whitespace()
                .tokenize()
                .chunk_split(max_length=chunk_size)
                .embed(method=method)
                .get_data()
            )
            self.query_embedding[idx] = embedded_query

    def search(self, query_idx: int, top_k: int=5):
        query = self.query_embedding[query_idx]
        distances, indices, result_texts, result_metadata = self.vector_store.similarity_search(query, k=top_k)
        return {
            'texts': result_texts,
            'distances': distances
        }
    
    def evaluate(self, top_k: int=100):
        results = {}
        # Search for each query
        for idx, _ in self.query_embedding.items():
            results[idx] = self.search(idx, top_k)
        # Evaluate results
        metrics = {
            '@5': {
                'ndcg': [],
                'recall': []
            },
            '@10': {
                'ndcg': [],
                'recall': []
            }
        }
        def metris_at_k(k, true_docs):
                ndcg = ndcg_at_k(_unique(results[idx]['texts'])[:k], true_docs, k)
                recall = len(set(_unique(results[idx]['texts'])[:k]) & set(true_docs)) / len(true_docs)
                return ndcg, recall
        for idx, _ in results.items():
            true_docs = self.qrels[self.qrels['query_id'] == idx]['corpus_id'].values # Get the true documents labels for the query
            ndcg_5, recall_5 = metris_at_k(5, true_docs)
            metrics['@5']['ndcg'].append(ndcg_5)
            metrics['@5']['recall'].append(recall_5)
            ndcg_10, recall_10 = metris_at_k(10, true_docs)
            metrics['@10']['ndcg'].append(ndcg_10)
            metrics['@10']['recall'].append(recall_10)
        return {
            'ndcg': {
                '@5': np.mean(metrics['@5']['ndcg']),
                '@10': np.mean(metrics['@10']['ndcg'])
            },
            'recall': {
                '@5': np.mean(metrics['@5']['recall']),
                '@10': np.mean(metrics['@10']['recall'])
            }
        }
            
def _unique(lst):
    _, idx = np.unique(lst, return_index=True)  # Get the unique elements and their first occurrence index
    sorted_idx = np.sort(idx)                  # Sort the indices to preserve the original order
    return [lst[i] for i in sorted_idx]        # Retrieve elements based on sorted indices

def ndcg_at_k(pred_docs, true_docs, k=10):
    y_score = np.array(list(range(1,k+1))[::-1])
    y_true = np.zeros(k, dtype=int)
    for i, doc in enumerate(pred_docs):
        if doc in true_docs:
            y_true[i] = 1
    return ndcg_score(np.asarray([y_true]), np.asarray([y_score]))
