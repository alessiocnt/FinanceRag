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
    
    # def embed_corpus(self, chunk_size: int=256, method: str=None):
    #     for idx, text in self.corpus.items():
    #         embedded_doc = (
    #             self.text_processor.load_data(text)
    #             # .to_lowercase()
    #             # .remove_numbers()
    #             # .remove_punctuation()
    #             # .remove_stopwords()
    #             # .lemmatize_text()
    #             # .remove_extra_whitespace()
    #             .remove_tables()
    #             .tokenize()
    #             .chunk_split(max_length=chunk_size)
    #             .embed(method=method)
    #             .get_data()
    #         )
    #         self.corpus_embedding[idx] = embedded_doc

    def manage_corpus(self, chunk_size: int=256, method: str=None, remove_tables: bool=False):    
        for idx, text in self.corpus.items():
            self.text_processor.load_data(text)
            if remove_tables:
                self.text_processor.remove_tables()
            embedded_doc = (
                self.text_processor.tokenize()
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
            'distances': distances, 
        }
    
    def retrieve(self, top_k: int=100):
        results = {}    # {query_id: {texts: [], distances: []}}
        # Search for each query
        for idx, _ in self.query_embedding.items():
            results[idx] = self.search(idx, top_k)
        return results

    def get_true_docs(self, query_id):
            return self.qrels[self.qrels['query_id'] == query_id]['corpus_id'].values

    def evaluate(self, results: Dict, k1: int=5, k2: int=10):
        # Evaluate results
        metrics = {
            f'@{k1}': {
                'ndcg': [],
                'recall': [], 
                'mrr': [],
                'map': []
            },
            f'@{k2}': {
                'ndcg': [],
                'recall': [],
                'mrr': [],
                'map': []
            }
        }

        def metris_at_k(k, true_docs, idx):
                ndcg = ndcg_at_k(_unique(results[idx]['texts'])[:k], true_docs, k)
                recall = len(set(_unique(results[idx]['texts'])[:k]) & set(true_docs)) / len(true_docs)
                mrr = mrr_by_hand(_unique(results[idx]['texts'])[:k], true_docs)
                map_ = map_by_hand(_unique(results[idx]['texts'])[:k], true_docs)
                return ndcg, recall, mrr, map_
        
        for idx, _ in results.items():
            true_docs = self.qrels[self.qrels['query_id'] == idx]['corpus_id'].values # Get the true documents labels for the query
            ndcg_k1, recall_k1, mrr_k1, map_k1 = metris_at_k(k1, true_docs, idx)
            metrics[f'@{k1}']['ndcg'].append(ndcg_k1)
            metrics[f'@{k1}']['recall'].append(recall_k1)
            metrics[f'@{k1}']['mrr'].append(mrr_k1)
            metrics[f'@{k1}']['map'].append(map_k1)
            ndcg_k2, recall_k2, mrr_k2, map_k2 = metris_at_k(k2, true_docs, idx)
            metrics[f'@{k2}']['ndcg'].append(ndcg_k2)
            metrics[f'@{k2}']['recall'].append(recall_k2)
            metrics[f'@{k2}']['mrr'].append(mrr_k2)
            metrics[f'@{k2}']['map'].append(map_k2)
        return {
            'ndcg': {
                f'@{k1}': float(round(np.mean(metrics[f'@{k1}']['ndcg']),4)),
                f'@{k2}': float(round(np.mean(metrics[f'@{k2}']['ndcg']),4))
            },
            'recall': {
                f'@{k1}': float(round(np.mean(metrics[f'@{k1}']['recall']),4)),
                f'@{k2}': float(round(np.mean(metrics[f'@{k2}']['recall']),4))
            },
            'mrr': {
                f'@{k1}': float(round(np.mean(metrics[f'@{k1}']['mrr']),4)),
                f'@{k2}': float(round(np.mean(metrics[f'@{k2}']['mrr']),4))
            },
            'map': {
                f'@{k1}': float(round(np.mean(metrics[f'@{k1}']['map']),4)),
                f'@{k2}': float(round(np.mean(metrics[f'@{k2}']['map']),4))
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

def ndcf_by_hand(y_true):
    DCG = 0
    for i, y in enumerate(y_true):
        DCG += (2**y - 1) / np.log2(i + 2)
    IDCG = 0
    # sort y_true in descending order
    y_true = sorted(y_true, reverse=True)
    for i, y in enumerate(y_true):
        IDCG += (2**y - 1) / np.log2(i + 2)
    return round(float(DCG / IDCG),4)

def mrr_by_hand(pred_docs, true_docs):
    mrr = 0
    for i, doc in enumerate(pred_docs):
        if doc in true_docs:
            mrr += 1 / (i + 1)
    return mrr / len(true_docs)


def map_by_hand(pred_docs, true_docs):
    true_docs = set(true_docs) 
    retrieved = 0
    precision_sum = 0
    for i, doc in enumerate(pred_docs):
        if doc in true_docs:
            retrieved += 1
            precision_at_rank = retrieved / (i + 1)  # Precision at rank i+1
            precision_sum += precision_at_rank
    if retrieved == 0: 
        return 0.0
    return precision_sum / len(true_docs)       
    #  precision = len(set(_unique(results[idx]['texts'])[:k]) & set(true_docs)) / k