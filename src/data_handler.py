import re
import string
from typing import List, Callable, Union
import nltk
# pip install -U spacy
# python -m spacy download en_core_web_sm
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from embeddings import *

# # Download required NLTK resources
# nltk.download('stopwords')
# nltk.download('punkt')

class DataHandler:
    def __init__(self, tokenizer=None, embedding_model=None, language='english'):
        """
        Initialize the text processor with raw data.
        :param data: List of text documents to be processed or a single string.
        :param language: Language for stopwords and lemmatization (default: 'en').
        """
        self.data = ""
        self.tokenizer = tokenizer if tokenizer else nltk.word_tokenize
        self.embedding_model = embedding_model if embedding_model else TfidfVectorizer()
        self.language = language
        self.nlp = spacy.load('en_core_web_sm')  # Load spaCy model for lemmatization
        self.stop_words = set(stopwords.words(language))
        self.stemmer = PorterStemmer()

    def load_data(self, data: Union[List[str], str]):
        """Load new data to be processed."""
        self.data = data if isinstance(data, list) else [data]
        return self

    def to_lowercase(self):
        """Convert all text to lowercase."""
        self.data = [text.lower() for text in self.data]
        return self

    def remove_punctuation(self):
        """Remove punctuation from the text."""
        self.data = [text.translate(str.maketrans('', '', string.punctuation)) for text in self.data]
        return self

    def remove_stopwords(self):
        """Remove common stopwords from the text."""
        self.data = [" ".join([word for word in text.split() if word not in self.stop_words]) for text in self.data]
        return self

    def remove_numbers(self):
        """Remove numbers from the text."""
        self.data = [re.sub(r'\d+', '', text) for text in self.data]
        return self

    def apply_custom_function(self, func: Callable[[str], str]):
        """Apply a custom function to transform the text."""
        self.data = [func(text) for text in self.data]
        return self

    def stem_text(self):
        """Apply stemming to reduce words to their root form."""
        self.data = [" ".join([self.stemmer.stem(word) for word in text.split()]) for text in self.data]
        return self

    def lemmatize_text(self):
        """Apply lemmatization using spaCy to get the base form of words."""
        self.data = [" ".join([token.lemma_ for token in self.nlp(text)]) for text in self.data]
        return self

    def remove_extra_whitespace(self):
        """Remove extra spaces and strip whitespace."""
        self.data = [" ".join(text.split()) for text in self.data]
        return self

    def tokenize(self):
        """Tokenize the text into words."""
        self.data = self.tokenizer.tokenize(self.data)
        return self
    
    def _chunk_split(self, text, max_length=512):
        """Split the text into chunks of a maximum length."""
        return chunker(self.data)

    def embed(self):
        """Convert text data into embeddings."""
        self.data = self.embedding_model.generate_embeddings(self._chunk_split(self.data))
        return self

    def get_data(self):
        """Return the processed data."""
        return self.data

def extract_tables(text):
    # Regex pattern to match table-like structures
    table_pattern = r'(?:(?:\n|\A)([^\n]+\|[^\n]+\n)((?:[-]+\|[-]+\n)?)((?:[^\n]+\|[^\n]+\n)+))'
    tables = []
    for match in re.finditer(table_pattern, text, re.MULTILINE):
        # Extract header and rows
        header = [h.strip() for h in match.group(1).strip().split('|')]
        rows = [row.strip().split('|') for row in match.group(3).strip().split('\n')]
        # Clean and strip each cell
        rows = [[cell.strip() for cell in row] for row in rows]
        # Create a text representation
        table_text = "Columns: " + " | ".join(header) + "\n"
        table_text += "Rows:\n"
        for row in rows:
            table_text += " | ".join(row) + "\n"
        tables.append(table_text)
    return tables