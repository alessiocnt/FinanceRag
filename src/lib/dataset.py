import os
import gzip
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

class FinanceRAGDataset:
    def __init__(self, data_dir: str):
        """
        Initialize the dataset manager with the path to the data directory.
        :param data_dir: Path to the main 'data' directory.
        """
        self.data_dir = data_dir
        self.datasets = self._discover_datasets()

    def _discover_datasets(self) -> Dict[str, Dict[str, str]]:
        """
        Discover available datasets and their files within the data directory.
        :return: Dictionary containing dataset names and their file paths.
        """
        datasets = {}
        for dataset_name in os.listdir(self.data_dir):
            dataset_path = os.path.join(self.data_dir, dataset_name)
            if os.path.isdir(dataset_path):
                datasets[dataset_name] = {
                    "corpus": os.path.join(dataset_path, "corpus.jsonl.gz"),
                    "queries": os.path.join(dataset_path, "queries.jsonl.gz"),
                    "qrels": os.path.join(dataset_path, "qrels.tsv")
                }
        return datasets

    def list_datasets(self) -> List[str]:
        """
        List all available datasets.
        :return: List of dataset names.
        """
        return list(self.datasets.keys())

    def load_corpus(self, dataset_name: str) -> List[Dict]:
        """
        Load the corpus.jsonl.gz for a specific dataset.
        :param dataset_name: Name of the dataset to load.
        :return: List of corpus data as dictionaries.
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found.")
        corpus_path = self.datasets[dataset_name]["corpus"]
        return self._load_jsonl_gz(corpus_path)

    def load_queries(self, dataset_name: str) -> List[Dict]:
        """
        Load the queries.jsonl.gz for a specific dataset.
        :param dataset_name: Name of the dataset to load.
        :return: List of query data as dictionaries.
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found.")
        queries_path = self.datasets[dataset_name]["queries"]
        return self._load_jsonl_gz(queries_path)

    def load_qrels(self, dataset_name: str) -> pd.DataFrame:
        """
        Load the qrels.tsv file containing the labels for a specific dataset.
        :param dataset_name: Name of the dataset to load.
        :return: Pandas DataFrame with the label data.
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found.")
        qrels_path = self.datasets[dataset_name]["qrels"]
        return self._load_tsv(qrels_path)
    
    def load_dataset(self, dataset_name: str) -> Tuple[List[Dict], List[Dict], pd.DataFrame]:
        """
        Load all data for a specific dataset.
        :param dataset_name: Name of the dataset to load.
        :return: Tuple of (corpus, queries, qrels)
        """
        return self.load_corpus(dataset_name), self.load_queries(dataset_name), self.load_qrels(dataset_name)

    def _load_jsonl_gz(self, file_path: str) -> List[Dict]:
        """
        Helper function to read a .jsonl.gz file and return the data.
        :param file_path: Path to the .jsonl.gz file.
        :return: List of parsed JSON objects.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' not found.")
        data = []
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

    def _load_tsv(self, file_path: str) -> pd.DataFrame:
        """
        Helper function to read a TSV file and return it as a DataFrame.
        :param file_path: Path to the .tsv file.
        :return: Pandas DataFrame of the TSV contents.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' not found.")

        return pd.read_csv(file_path, sep='\t')

# Reduce the dataset size to the only document usefull for testing pourpose. Only 30% of labels are available. We focus only on those corpus.
def reduce_dataset_size(corpus: List[Dict], 
                        queries: List[Dict],
                        qrels: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
    """
    Reduce the dataset size to a specific ratio based on the available labels.
    :param corpus: List of corpus data as dictionaries.
    :param queries: List of query data as dictionaries.
    :param qrels: Pandas DataFrame with the label data.
    :return: Tuple of (reduced corpus, reduced queries)
    """
    # Filter corpus
    doc_ids = np.array([elem["_id"] for elem in corpus])
    doc_texts = np.array([elem["text"] for elem in corpus])
    mask_queries = np.isin(doc_ids, np.unique(qrels["corpus_id"]))
    doc_ids = doc_ids[mask_queries]
    doc_texts = [[doc] for doc in doc_texts[mask_queries]]
    corpus = dict(zip(doc_ids, doc_texts))
    # Filter queries
    query_ids = np.array([elem["_id"] for elem in queries])
    query_texts = np.array([elem["text"] for elem in queries])
    mask_queries = np.isin(query_ids, np.unique(qrels["query_id"]))
    query_ids = query_ids[mask_queries]
    query_texts = query_texts[mask_queries]
    queries = dict(zip(query_ids, query_texts))
    return corpus, queries