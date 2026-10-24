import nltk
from rank_bm25 import BM25Okapi

# nltk >= 3.9 needs punkt_tab for word_tokenize; older data layouts only
# ship punkt, so ensure both before tokenizing.
for _resource in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{_resource}")
    except LookupError:
        nltk.download(_resource)


class KeywordSearchBM25:
    def __init__(self, documents, metadatas=None):
        """
        Initialize the KeywordSearchBM25 class with a list of documents.

        Parameters:
        - documents (list of str): A list of documents to be searched.
        - metadatas (list of dict, optional): Metadata for each document,
          same length/order as `documents`. Defaults to empty dicts so
          metadata always flows alongside text, matching what's stored
          in Pinecone for the same chunk.
        """
        self.documents = documents
        self.metadatas = metadatas if metadatas is not None else [{} for _ in documents]
        self.tokenized_docs = [nltk.word_tokenize(doc.lower()) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def preprocess_query(self, query):
        """
        Preprocess the query by tokenizing and lowercasing.

        Parameters:
        - query (str): The search query.

        Returns:
        - list of str: Tokenized query.
        """
        return nltk.word_tokenize(query.lower())

    def search(self, query, top_n=5):
        """
        Perform a keyword-based search using BM25 and return the top N results.

        Parameters:
        - query (str): The search query.
        - top_n (int): The number of top results to return.

        Returns:
        - list of tuple: (score, document text, metadata dict) for the top N results.
        """
        tokenized_query = self.preprocess_query(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            zip(scores, self.documents, self.metadatas, strict=False),
            key=lambda x: x[0],
            reverse=True,
        )[:top_n]
        return [(score, doc, meta) for score, doc, meta in ranked]


# Example usage
if __name__ == "__main__":
    documents = [
        """Machine learning is a field of artificial intelligence that uses statistical
        techniques to give computer systems the ability to learn from data, without
        being explicitly programmed. This includes supervised learning, unsupervised
        learning, and reinforcement learning.""",
        """Natural language processing (NLP) is a branch of artificial intelligence focused on the
        interaction between computers and humans through natural language. NLP includes tasks like
        language translation, sentiment analysis, and speech recognition.""",
        """BM25 is an algorithm used in information retrieval that ranks documents based on their
        relevance to a given search query. It is often applied in keyword-based search systems
        and is particularly effective in ranking large collections of text data.""",
        """Hybrid search combines the strengths of keyword-based search with semantic search.
        Keyword-based search like BM25 focuses on matching exact terms, while semantic search
        uses dense vector representations to capture the contextual meaning of terms in a query.""",
        """Deep learning has revolutionized computer vision, enabling applications like image
        classification, object detection, and facial recognition. Convolutional neural networks
        (CNNs) are a common architecture used in these applications, designed specifically to
        process grid-like data such as images.""",
    ]

    keyword_search = KeywordSearchBM25(documents)
    query = "artificial intelligence and machine learning"
    results = keyword_search.search(query, top_n=3)

    for _score, _doc, _meta in results:
        pass
