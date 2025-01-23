import pandas as pd
import numpy as np
from dataset import *
from transformers import AutoTokenizer, AutoModel
from embeddings import EmbeddingGenerator
import faiss


EMBEDDING_DIM = 768



if __name__ == "__main__":
    dataset_manager = FinanceRAGDataset("./data")

    # Load a dataset between: ConvFinQA, FinQA, MultiHeritt, and TATQA
    dataset_name = "ConvFinQA"      
    corpus, queries, labels = dataset_manager.load_dataset(dataset_name)

    corpus_df = pd.DataFrame(corpus)
    queries_df = pd.DataFrame(queries)
    corpus_df.drop(columns=["title"], inplace=True)
    queries_df.drop(columns=["title"], inplace=True)


    # Preprocessing del corpus: 
    # lowercasing
    # rimozione simboli, punteggiatura(?)
    # rimozione di stopwords (potrebbe togliere informazioni utili)
    # stemming e lemmatization
    # Fare step di preprocessing anche sulle queries!



    # Identificazione delle tabelle per ogni corpus
    # Processamento con LLM delle tabelle
    # Creazione dei chunk per il restante testo (vedere che si può intendere con "creazione")
    # vedere vector store



    # Creazione degli embeddings: della query, dei chunks e del summary delle tabelle
    # Salvataggio degli embeddings in un file .npy, e sucessivamente caricarli 

    # Load FinBERT model for tokenization and embeddings 
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-pretrain")
    model = AutoModel.from_pretrained("yiyanghkust/finbert-pretrain")

    if os.path.exists("Embeddings/corpus_embeddings.npy"):
        corpus_embeddings = np.load("Embeddings/corpus_embeddings.npy")
    else:
        emb_generator = EmbeddingGenerator(tokenizer=tokenizer, model=model, method=2, chunck_size=16)
        # testing a generare un embedding
        emb = emb_generator.generate_single_embedding(corpus_df["text"].values[0])
        print(emb.shape)
        np.save("Embeddings/prove_emb.npy", emb)




    # Creazione dell'indice FAISS per la ricerca dei documenti in maniera efficiente
    index = faiss.IndexFlatL2(EMBEDDING_DIM)    # Similarità L2
    index.add(corpus_embeddings)
    # Salva l'indice FAISS
    faiss.write_index(index, "faiss_index.bin")
    # Ricarica l'indice FAISS
    # index = faiss.read_index("faiss_index.bin")


    # Ricerca dei documenti più rilevanti per una query con l'indice FAISS