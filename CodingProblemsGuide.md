# GenAI/RAG Coding Problems Guide

Actual coding problems you might face in interviews for GenAI/LLM/RAG positions.

---

## Problem 1: Implement Cosine Similarity

**Question:** Implement cosine similarity from scratch without using libraries.

```python
def cosine_similarity(vec1: list, vec2: list) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score between -1 and 1
    """
    # Your implementation here
    pass

# Test
vec1 = [1, 2, 3]
vec2 = [4, 5, 6]
print(cosine_similarity(vec1, vec2))  # Expected: ~0.974
```

**Solution:**
```python
import math

def cosine_similarity(vec1: list, vec2: list) -> float:
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have same length")

    # Dot product
    dot_product = sum(a * b for a, b in zip(vec1, vec2))

    # Magnitudes
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))

    # Handle zero vectors
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)

# Test
vec1 = [1, 2, 3]
vec2 = [4, 5, 6]
print(cosine_similarity(vec1, vec2))  # 0.9746318461970762
```

---

## Problem 2: Implement Simple Text Chunking

**Question:** Implement a text chunker that splits text into chunks with overlap.

```python
def chunk_text(text: str, chunk_size: int, overlap: int) -> list:
    """
    Split text into overlapping chunks.

    Args:
        text: Input text
        chunk_size: Maximum characters per chunk
        overlap: Number of overlapping characters

    Returns:
        List of text chunks
    """
    pass

# Test
text = "The quick brown fox jumps over the lazy dog. It was a sunny day."
chunks = chunk_text(text, chunk_size=30, overlap=10)
print(chunks)
```

**Solution:**
```python
def chunk_text(text: str, chunk_size: int, overlap: int) -> list:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        # Move start position
        start = end - overlap

        # If remaining text is smaller than overlap, break
        if start >= len(text):
            break

    return chunks

# Better version with word boundaries
def chunk_text_smart(text: str, chunk_size: int, overlap: int) -> list:
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        word_length = len(word) + 1  # +1 for space

        if current_length + word_length > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))

            # Keep overlap words
            overlap_words = []
            overlap_length = 0
            for w in reversed(current_chunk):
                if overlap_length + len(w) + 1 <= overlap:
                    overlap_words.insert(0, w)
                    overlap_length += len(w) + 1
                else:
                    break

            current_chunk = overlap_words
            current_length = overlap_length

        current_chunk.append(word)
        current_length += word_length

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks
```

---

## Problem 3: Implement K-Nearest Neighbors Search

**Question:** Implement brute-force k-NN search for vectors.

```python
def knn_search(query: list, vectors: list, k: int) -> list:
    """
    Find k nearest neighbors to query vector.

    Args:
        query: Query vector
        vectors: List of (id, vector) tuples
        k: Number of neighbors

    Returns:
        List of (id, distance) tuples, sorted by distance
    """
    pass

# Test
vectors = [
    ("doc1", [1, 0, 0]),
    ("doc2", [0, 1, 0]),
    ("doc3", [1, 1, 0]),
    ("doc4", [0, 0, 1]),
]
query = [1, 0.5, 0]
results = knn_search(query, vectors, k=2)
print(results)  # Should return doc1 and doc3
```

**Solution:**
```python
import math

def euclidean_distance(vec1, vec2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))

def knn_search(query: list, vectors: list, k: int) -> list:
    # Calculate distances
    distances = []
    for doc_id, vec in vectors:
        dist = euclidean_distance(query, vec)
        distances.append((doc_id, dist))

    # Sort by distance
    distances.sort(key=lambda x: x[1])

    # Return top k
    return distances[:k]

# Using cosine similarity (more common for text)
def knn_search_cosine(query: list, vectors: list, k: int) -> list:
    similarities = []
    for doc_id, vec in vectors:
        sim = cosine_similarity(query, vec)
        similarities.append((doc_id, sim))

    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:k]
```

---

## Problem 4: Implement TF-IDF

**Question:** Implement TF-IDF scoring from scratch.

```python
def compute_tfidf(documents: list) -> dict:
    """
    Compute TF-IDF scores for all terms in documents.

    Args:
        documents: List of document strings

    Returns:
        Dict mapping doc_index to {term: tfidf_score}
    """
    pass

# Test
documents = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are pets"
]
tfidf = compute_tfidf(documents)
```

**Solution:**
```python
import math
from collections import Counter

def compute_tfidf(documents: list) -> dict:
    # Tokenize
    tokenized_docs = [doc.lower().split() for doc in documents]
    n_docs = len(documents)

    # Compute document frequency (DF)
    df = Counter()
    for doc in tokenized_docs:
        unique_terms = set(doc)
        for term in unique_terms:
            df[term] += 1

    # Compute TF-IDF for each document
    tfidf_scores = {}

    for doc_idx, tokens in enumerate(tokenized_docs):
        # Term frequency
        tf = Counter(tokens)
        total_terms = len(tokens)

        doc_tfidf = {}
        for term, count in tf.items():
            # TF: count / total terms in doc
            tf_score = count / total_terms

            # IDF: log(N / df)
            idf_score = math.log(n_docs / df[term])

            # TF-IDF
            doc_tfidf[term] = tf_score * idf_score

        tfidf_scores[doc_idx] = doc_tfidf

    return tfidf_scores

# Test
documents = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are pets"
]
tfidf = compute_tfidf(documents)
for doc_idx, scores in tfidf.items():
    print(f"Doc {doc_idx}: {scores}")
```

---

## Problem 5: Implement BM25 Retrieval

**Question:** Implement BM25 ranking algorithm.

```python
def bm25_search(query: str, documents: list, k1: float = 1.5, b: float = 0.75) -> list:
    """
    BM25 ranking for documents.

    Args:
        query: Search query
        documents: List of document strings
        k1: Term frequency saturation parameter
        b: Length normalization parameter

    Returns:
        List of (doc_index, score) sorted by score
    """
    pass
```

**Solution:**
```python
import math
from collections import Counter

def bm25_search(query: str, documents: list, k1: float = 1.5, b: float = 0.75) -> list:
    # Tokenize
    query_terms = query.lower().split()
    tokenized_docs = [doc.lower().split() for doc in documents]
    n_docs = len(documents)

    # Calculate average document length
    avg_doc_len = sum(len(doc) for doc in tokenized_docs) / n_docs

    # Calculate document frequency
    df = Counter()
    for doc in tokenized_docs:
        for term in set(doc):
            df[term] += 1

    # Calculate BM25 scores
    scores = []

    for doc_idx, doc_tokens in enumerate(tokenized_docs):
        score = 0
        doc_len = len(doc_tokens)
        tf = Counter(doc_tokens)

        for term in query_terms:
            if term not in tf:
                continue

            # IDF component
            idf = math.log((n_docs - df[term] + 0.5) / (df[term] + 0.5) + 1)

            # TF component with saturation
            term_freq = tf[term]
            tf_component = (term_freq * (k1 + 1)) / (
                term_freq + k1 * (1 - b + b * (doc_len / avg_doc_len))
            )

            score += idf * tf_component

        scores.append((doc_idx, score))

    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

# Test
documents = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are pets",
    "the cat chased the dog"
]
results = bm25_search("cat dog", documents)
print(results)
```

---

## Problem 6: Implement Simple RAG Pipeline

**Question:** Build a basic RAG pipeline with in-memory vector store.

```python
class SimpleRAG:
    def __init__(self, documents: list):
        """Initialize with documents."""
        pass

    def add_document(self, doc: str):
        """Add a document to the knowledge base."""
        pass

    def query(self, question: str, k: int = 3) -> str:
        """Answer question using RAG."""
        pass
```

**Solution:**
```python
import numpy as np
from sentence_transformers import SentenceTransformer

class SimpleRAG:
    def __init__(self, documents: list = None):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.embeddings = []

        if documents:
            for doc in documents:
                self.add_document(doc)

    def add_document(self, doc: str):
        """Add document and compute embedding."""
        self.documents.append(doc)
        embedding = self.encoder.encode(doc)
        self.embeddings.append(embedding)

    def retrieve(self, query: str, k: int = 3) -> list:
        """Retrieve top-k relevant documents."""
        query_embedding = self.encoder.encode(query)

        # Calculate similarities
        similarities = []
        for i, emb in enumerate(self.embeddings):
            sim = np.dot(query_embedding, emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(emb)
            )
            similarities.append((i, sim))

        # Sort and get top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:k]

        return [self.documents[i] for i, _ in top_k]

    def query(self, question: str, k: int = 3) -> str:
        """Full RAG: retrieve + generate."""
        # Retrieve relevant documents
        relevant_docs = self.retrieve(question, k)
        context = "\n".join(relevant_docs)

        # In real implementation, call LLM here
        # For demo, return formatted context
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""

        # Here you would call: llm.invoke(prompt)
        return prompt

# Test
rag = SimpleRAG([
    "Python is a programming language created by Guido van Rossum.",
    "Python is known for its simple and readable syntax.",
    "Machine learning is a subset of artificial intelligence.",
    "Python is widely used in data science and AI."
])

result = rag.query("What is Python used for?")
print(result)
```

---

## Problem 7: Implement Conversation Memory

**Question:** Implement a conversation memory with sliding window.

```python
class ConversationMemory:
    def __init__(self, max_turns: int = 5):
        """Initialize with max conversation turns to remember."""
        pass

    def add_message(self, role: str, content: str):
        """Add a message to memory."""
        pass

    def get_context(self) -> str:
        """Get conversation context as string."""
        pass

    def clear(self):
        """Clear memory."""
        pass
```

**Solution:**
```python
from collections import deque
from datetime import datetime

class ConversationMemory:
    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.messages = deque(maxlen=max_turns * 2)  # User + Assistant per turn

    def add_message(self, role: str, content: str):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

    def get_messages(self) -> list:
        return list(self.messages)

    def get_context(self) -> str:
        """Format conversation as string."""
        lines = []
        for msg in self.messages:
            role = msg["role"].capitalize()
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    def get_messages_for_api(self) -> list:
        """Format for OpenAI API."""
        return [{"role": m["role"], "content": m["content"]} for m in self.messages]

    def clear(self):
        self.messages.clear()

# Test
memory = ConversationMemory(max_turns=3)
memory.add_message("user", "Hi, my name is Alice")
memory.add_message("assistant", "Hello Alice! How can I help you?")
memory.add_message("user", "What's my name?")
memory.add_message("assistant", "Your name is Alice.")
memory.add_message("user", "Tell me about Python")
memory.add_message("assistant", "Python is a programming language.")
memory.add_message("user", "What's my name again?")

print(memory.get_context())
# Should only show last 3 turns (6 messages)
```

---

## Problem 8: Implement Semantic Caching

**Question:** Build a cache that returns similar queries' results.

```python
class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.9):
        """Cache with semantic similarity matching."""
        pass

    def get(self, query: str) -> str | None:
        """Get cached response if similar query exists."""
        pass

    def set(self, query: str, response: str):
        """Cache a query-response pair."""
        pass
```

**Solution:**
```python
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.9):
        self.threshold = similarity_threshold
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.cache = []  # List of (query, embedding, response)

    def _get_embedding(self, text: str):
        return self.encoder.encode(text)

    def _cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get(self, query: str) -> str | None:
        if not self.cache:
            return None

        query_emb = self._get_embedding(query)

        best_match = None
        best_similarity = 0

        for cached_query, cached_emb, response in self.cache:
            similarity = self._cosine_similarity(query_emb, cached_emb)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = response

        if best_similarity >= self.threshold:
            return best_match
        return None

    def set(self, query: str, response: str):
        embedding = self._get_embedding(query)
        self.cache.append((query, embedding, response))

# Test
cache = SemanticCache(similarity_threshold=0.85)
cache.set("What is Python?", "Python is a programming language.")
cache.set("How do I learn coding?", "Start with tutorials and practice.")

# Should hit cache (similar to "What is Python?")
print(cache.get("Tell me about Python"))  # Returns cached response

# Should miss cache
print(cache.get("What is the weather?"))  # Returns None
```

---

## Problem 9: Implement Re-ranking

**Question:** Implement a simple re-ranker using cross-encoder logic.

```python
def rerank_documents(query: str, documents: list, top_k: int = 3) -> list:
    """
    Re-rank documents based on relevance to query.

    Args:
        query: Search query
        documents: List of document strings
        top_k: Number of top documents to return

    Returns:
        Re-ranked list of (doc, score) tuples
    """
    pass
```

**Solution:**
```python
from sentence_transformers import CrossEncoder

def rerank_documents(query: str, documents: list, top_k: int = 3) -> list:
    """Re-rank using cross-encoder."""
    # Cross-encoder considers query-document pair together
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # Create query-document pairs
    pairs = [[query, doc] for doc in documents]

    # Get relevance scores
    scores = model.predict(pairs)

    # Combine and sort
    doc_scores = list(zip(documents, scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True)

    return doc_scores[:top_k]

# Simple version without cross-encoder (using term overlap)
def rerank_simple(query: str, documents: list, top_k: int = 3) -> list:
    """Simple re-ranking using term overlap and position."""
    query_terms = set(query.lower().split())

    scores = []
    for doc in documents:
        doc_lower = doc.lower()
        doc_terms = set(doc_lower.split())

        # Term overlap score
        overlap = len(query_terms & doc_terms) / len(query_terms) if query_terms else 0

        # Position bonus (query terms appearing early)
        position_score = 0
        for term in query_terms:
            pos = doc_lower.find(term)
            if pos != -1:
                position_score += 1 / (pos + 1)

        total_score = overlap * 0.7 + position_score * 0.3
        scores.append((doc, total_score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]
```

---

## Problem 10: Implement Token Counter

**Question:** Estimate token count for text (approximation).

```python
def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Estimate token count for text.
    For interview, implement approximation without tiktoken.
    """
    pass
```

**Solution:**
```python
# Using tiktoken (production)
import tiktoken

def count_tokens_tiktoken(text: str, model: str = "gpt-3.5-turbo") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Approximation (for interview without library)
def count_tokens_approx(text: str) -> int:
    """
    Rough approximation: ~4 characters per token for English
    Or ~0.75 words per token
    """
    # Method 1: Character-based
    char_estimate = len(text) / 4

    # Method 2: Word-based
    word_estimate = len(text.split()) / 0.75

    # Average both methods
    return int((char_estimate + word_estimate) / 2)

# Better approximation considering patterns
def count_tokens_better(text: str) -> int:
    """More accurate approximation."""
    count = 0

    # Split into words
    words = text.split()

    for word in words:
        if len(word) <= 4:
            count += 1  # Short words = 1 token
        elif len(word) <= 8:
            count += 1.5  # Medium words ≈ 1-2 tokens
        else:
            count += len(word) / 4  # Long words = multiple tokens

    # Add tokens for spaces and punctuation
    count += text.count(' ') * 0.1
    count += sum(text.count(p) for p in '.,!?;:') * 0.5

    return int(count)

# Test
text = "Hello, how are you doing today? I hope you're having a wonderful day!"
print(f"Approximate: {count_tokens_approx(text)}")
print(f"Better estimate: {count_tokens_better(text)}")
# print(f"Actual (tiktoken): {count_tokens_tiktoken(text)}")
```

---

## Problem 11: Implement Prompt Template Engine

**Question:** Build a simple prompt template system.

```python
class PromptTemplate:
    def __init__(self, template: str):
        """Initialize with template containing {variable} placeholders."""
        pass

    def format(self, **kwargs) -> str:
        """Format template with provided variables."""
        pass

    def get_variables(self) -> list:
        """Get list of variable names in template."""
        pass
```

**Solution:**
```python
import re

class PromptTemplate:
    def __init__(self, template: str):
        self.template = template
        self._variables = self._extract_variables()

    def _extract_variables(self) -> list:
        """Extract variable names from template."""
        pattern = r'\{(\w+)\}'
        return list(set(re.findall(pattern, self.template)))

    def format(self, **kwargs) -> str:
        """Format template with variables."""
        # Check all variables are provided
        missing = set(self._variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing variables: {missing}")

        result = self.template
        for key, value in kwargs.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result

    def get_variables(self) -> list:
        return self._variables.copy()

    def partial(self, **kwargs) -> 'PromptTemplate':
        """Create new template with some variables filled."""
        new_template = self.template
        for key, value in kwargs.items():
            new_template = new_template.replace(f"{{{key}}}", str(value))
        return PromptTemplate(new_template)

# Test
template = PromptTemplate("""
You are a {role} assistant.
Answer the following question about {topic}:
{question}
""")

print(template.get_variables())  # ['role', 'topic', 'question']

prompt = template.format(
    role="helpful",
    topic="Python",
    question="What are decorators?"
)
print(prompt)
```

---

## Problem 12: Implement Document Similarity Matrix

**Question:** Build a document similarity matrix.

```python
def build_similarity_matrix(documents: list) -> list:
    """
    Build NxN similarity matrix for documents.

    Returns:
        2D list where matrix[i][j] is similarity between doc i and j
    """
    pass
```

**Solution:**
```python
from sentence_transformers import SentenceTransformer
import numpy as np

def build_similarity_matrix(documents: list) -> list:
    # Get embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(documents)

    # Build matrix
    n = len(documents)
    matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 1.0
            else:
                # Cosine similarity
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                matrix[i][j] = float(sim)

    return matrix

# Efficient version using matrix multiplication
def build_similarity_matrix_fast(documents: list) -> np.ndarray:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(documents, normalize_embeddings=True)

    # Cosine similarity = dot product of normalized vectors
    similarity_matrix = np.dot(embeddings, embeddings.T)
    return similarity_matrix

# Test
docs = [
    "Python is a programming language",
    "Java is also a programming language",
    "Cats are cute animals",
    "Dogs are loyal pets"
]
matrix = build_similarity_matrix(docs)
for row in matrix:
    print([f"{x:.2f}" for x in row])
```

---

## Quick Reference: Common Interview Patterns

| Problem Type | Key Concepts |
|--------------|--------------|
| **Similarity** | Cosine similarity, dot product, Euclidean distance |
| **Retrieval** | K-NN, BM25, TF-IDF, inverted index |
| **Chunking** | Sliding window, overlap, recursive splitting |
| **Caching** | Semantic similarity threshold, LRU, TTL |
| **Memory** | Sliding window, summarization, token limits |
| **Ranking** | Bi-encoder vs cross-encoder, score normalization |

---

## Tips for Coding Interviews

1. **Clarify requirements** - Ask about edge cases, expected scale
2. **Start simple** - Get working solution first, optimize later
3. **Explain trade-offs** - Memory vs speed, accuracy vs latency
4. **Test your code** - Write test cases, walk through examples
5. **Know the libraries** - But be ready to implement from scratch
