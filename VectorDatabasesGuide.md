# Vector Databases Complete Guide

Everything you need to know about vector databases for interviews - comparisons, usage, and when to use what.

---

## What is a Vector Database?

A vector database is a specialized database designed to store, index, and query high-dimensional vectors (embeddings).

```
Traditional DB: Exact match on structured data
Vector DB: Similarity search on unstructured data (text, images, audio)
```

---

## Why Vector Databases?

| Problem | Traditional DB | Vector DB |
|---------|---------------|-----------|
| "Find similar products" | Can't do semantically | ✓ Similarity search |
| "Search by meaning" | Keyword only | ✓ Semantic search |
| "Find related documents" | Manual tagging | ✓ Automatic via embeddings |
| Scale to millions of vectors | Slow | ✓ Optimized indexes |

---

## How Vector Search Works

### 1. Embedding Generation
```
Text: "I love machine learning"
    ↓ Embedding Model
Vector: [0.12, -0.34, 0.56, ..., 0.78]  # 384-1536 dimensions
```

### 2. Indexing
```
Vectors stored in optimized data structures:
- Flat: Exact search (slow, accurate)
- IVF: Inverted file index (clusters)
- HNSW: Hierarchical navigable small world (graph)
- PQ: Product quantization (compressed)
```

### 3. Similarity Search
```
Query: "ML is awesome"
    ↓ Same embedding model
Query Vector: [0.11, -0.32, 0.58, ..., 0.75]
    ↓ Similarity calculation
Find top-k nearest neighbors
```

---

## Similarity Metrics

### Cosine Similarity (Most Common)
```
Measures angle between vectors
Range: -1 to 1 (1 = identical direction)
Best for: Normalized embeddings, text similarity

Formula: cos(θ) = (A·B) / (||A|| × ||B||)
```

### Euclidean Distance (L2)
```
Measures straight-line distance
Range: 0 to ∞ (0 = identical)
Best for: When magnitude matters

Formula: d = √Σ(ai - bi)²
```

### Dot Product (Inner Product)
```
Measures both angle and magnitude
Range: -∞ to ∞
Best for: Maximum inner product search (MIPS)

Formula: A·B = Σ(ai × bi)
```

### When to Use Which?

| Metric | Use Case |
|--------|----------|
| **Cosine** | Text embeddings (OpenAI, sentence-transformers) |
| **Euclidean** | Image embeddings, when scale matters |
| **Dot Product** | Recommendation systems, MIPS |

---

## Index Types Explained

### 1. Flat Index (Brute Force)
```
✓ 100% accurate (exact search)
✗ Slow for large datasets
✗ O(n) search time

Best for: < 10,000 vectors, when accuracy is critical
```

### 2. IVF (Inverted File Index)
```
How: Clusters vectors, searches only relevant clusters
✓ Faster than flat
✗ May miss some results

Parameters:
- nlist: Number of clusters (√n typical)
- nprobe: Clusters to search (trade-off speed/accuracy)

Best for: Medium datasets (10K - 1M vectors)
```

### 3. HNSW (Hierarchical Navigable Small World)
```
How: Graph-based, navigates through layers
✓ Very fast search
✓ Good accuracy
✗ High memory usage
✗ Slow to build

Parameters:
- M: Connections per node (16-64)
- ef_construction: Build quality (100-500)
- ef_search: Search quality (trade-off speed/accuracy)

Best for: Production systems, need speed + accuracy
```

### 4. PQ (Product Quantization)
```
How: Compresses vectors into codes
✓ Low memory
✗ Lower accuracy

Best for: Very large datasets, memory constrained
```

### 5. Hybrid: IVF-PQ, HNSW-PQ
```
Combines clustering/graph with compression
Balance between speed, accuracy, and memory
```

---

## Vector Database Comparison

### Quick Comparison Table

| Database | Type | Best For | Scaling | Cost |
|----------|------|----------|---------|------|
| **ChromaDB** | Embedded | Prototypes, small projects | Single node | Free |
| **FAISS** | Library | Research, custom solutions | Single node | Free |
| **Pinecone** | Cloud | Production, managed | Auto-scaling | $$ |
| **Weaviate** | Self-hosted/Cloud | Hybrid search, GraphQL | Horizontal | Free/$ |
| **Qdrant** | Self-hosted/Cloud | High performance | Horizontal | Free/$ |
| **Milvus** | Self-hosted | Enterprise, large scale | Distributed | Free |
| **pgvector** | PostgreSQL ext | Existing Postgres users | Postgres limits | Free |

---

## ChromaDB

### Overview
- **Type**: Embedded (in-process) or client-server
- **Best for**: Prototyping, small-medium projects
- **Index**: HNSW (via hnswlib)

### Installation
```bash
pip install chromadb
```

### Basic Usage
```python
import chromadb

# Create client
client = chromadb.Client()  # In-memory
client = chromadb.PersistentClient(path="./chroma_db")  # Persistent

# Create collection
collection = client.create_collection(
    name="my_collection",
    metadata={"hnsw:space": "cosine"}  # cosine, l2, ip
)

# Add documents
collection.add(
    documents=["doc1 text", "doc2 text", "doc3 text"],
    metadatas=[{"source": "a"}, {"source": "b"}, {"source": "c"}],
    ids=["id1", "id2", "id3"]
)

# Add with embeddings (if you have them)
collection.add(
    embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
    documents=["doc1", "doc2"],
    ids=["id1", "id2"]
)

# Query
results = collection.query(
    query_texts=["search query"],
    n_results=5,
    where={"source": "a"},  # Metadata filter
    include=["documents", "distances", "metadatas"]
)

# Get by ID
docs = collection.get(ids=["id1", "id2"])

# Update
collection.update(
    ids=["id1"],
    documents=["updated text"],
    metadatas=[{"source": "updated"}]
)

# Delete
collection.delete(ids=["id1"])
collection.delete(where={"source": "a"})
```

### With LangChain
```python
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="my_collection"
)

# Load existing
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="my_collection"
)

# Search
results = vectorstore.similarity_search("query", k=5)
results = vectorstore.similarity_search_with_score("query", k=5)
results = vectorstore.max_marginal_relevance_search("query", k=5)
```

### Pros & Cons
```
✓ Zero configuration
✓ Embedded (no separate server)
✓ Good for prototyping
✓ Automatic embedding (built-in models)

✗ Single node only
✗ Limited scalability
✗ Not for production at scale
```

---

## FAISS (Facebook AI Similarity Search)

### Overview
- **Type**: Library (not a database)
- **Best for**: Research, custom solutions, high performance
- **Indexes**: Flat, IVF, HNSW, PQ, and combinations

### Installation
```bash
pip install faiss-cpu  # CPU version
pip install faiss-gpu  # GPU version (CUDA required)
```

### Basic Usage
```python
import faiss
import numpy as np

# Create vectors
dimension = 384
vectors = np.random.random((10000, dimension)).astype('float32')

# === Flat Index (Exact) ===
index = faiss.IndexFlatL2(dimension)  # L2 distance
index = faiss.IndexFlatIP(dimension)  # Inner product
index.add(vectors)

# Search
query = np.random.random((1, dimension)).astype('float32')
distances, indices = index.search(query, k=5)

# === IVF Index ===
nlist = 100  # Number of clusters
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

# Must train before adding
index.train(vectors)
index.add(vectors)

# Search with nprobe
index.nprobe = 10  # Search 10 clusters
distances, indices = index.search(query, k=5)

# === HNSW Index ===
M = 32  # Connections per node
index = faiss.IndexHNSWFlat(dimension, M)
index.hnsw.efConstruction = 200  # Build quality
index.add(vectors)

index.hnsw.efSearch = 100  # Search quality
distances, indices = index.search(query, k=5)

# === Save and Load ===
faiss.write_index(index, "my_index.faiss")
index = faiss.read_index("my_index.faiss")
```

### With IDs
```python
# FAISS doesn't store IDs by default, use IndexIDMap
index = faiss.IndexFlatL2(dimension)
index_with_ids = faiss.IndexIDMap(index)

ids = np.array([100, 200, 300, 400, 500])
vectors = np.random.random((5, dimension)).astype('float32')
index_with_ids.add_with_ids(vectors, ids)

distances, indices = index_with_ids.search(query, k=3)
# indices now returns your custom IDs
```

### GPU Acceleration
```python
import faiss

# Move to GPU
gpu_resource = faiss.StandardGpuResources()
index_gpu = faiss.index_cpu_to_gpu(gpu_resource, 0, index)

# Search on GPU
distances, indices = index_gpu.search(query, k=5)
```

### With LangChain
```python
from langchain_community.vectorstores import FAISS

# Create
vectorstore = FAISS.from_documents(docs, embeddings)

# Save
vectorstore.save_local("faiss_index")

# Load
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Merge indexes
vectorstore.merge_from(other_vectorstore)
```

### Pros & Cons
```
✓ Extremely fast
✓ GPU support
✓ Many index options
✓ Battle-tested (by Meta)

✗ No built-in persistence (manual save/load)
✗ No metadata filtering
✗ No CRUD operations (rebuild to update)
✗ Library, not a database
```

---

## Pinecone

### Overview
- **Type**: Fully managed cloud service
- **Best for**: Production, zero-ops
- **Index**: Proprietary (optimized)

### Installation
```bash
pip install pinecone-client
```

### Basic Usage
```python
from pinecone import Pinecone, ServerlessSpec

# Initialize
pc = Pinecone(api_key="xxxxx")

# Create index
pc.create_index(
    name="my-index",
    dimension=384,
    metric="cosine",  # cosine, euclidean, dotproduct
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

# Get index
index = pc.Index("my-index")

# Upsert vectors
index.upsert(
    vectors=[
        {
            "id": "vec1",
            "values": [0.1, 0.2, ...],
            "metadata": {"genre": "comedy", "year": 2020}
        },
        {
            "id": "vec2",
            "values": [0.3, 0.4, ...],
            "metadata": {"genre": "drama", "year": 2021}
        }
    ],
    namespace="movies"  # Optional namespace
)

# Query
results = index.query(
    vector=[0.1, 0.2, ...],
    top_k=5,
    include_metadata=True,
    include_values=False,
    filter={
        "genre": {"$eq": "comedy"},
        "year": {"$gte": 2020}
    },
    namespace="movies"
)

# Fetch by ID
results = index.fetch(ids=["vec1", "vec2"])

# Delete
index.delete(ids=["vec1", "vec2"])
index.delete(filter={"genre": "comedy"})
index.delete(delete_all=True, namespace="movies")

# Get stats
stats = index.describe_index_stats()
```

### Metadata Filtering
```python
# Supported operators
filter = {
    "genre": {"$eq": "comedy"},        # Equals
    "year": {"$ne": 2020},             # Not equals
    "year": {"$gt": 2019},             # Greater than
    "year": {"$gte": 2020},            # Greater than or equal
    "year": {"$lt": 2022},             # Less than
    "year": {"$lte": 2021},            # Less than or equal
    "genre": {"$in": ["comedy", "drama"]},    # In list
    "genre": {"$nin": ["horror"]},     # Not in list

    # Combine with $and, $or
    "$and": [
        {"genre": {"$eq": "comedy"}},
        {"year": {"$gte": 2020}}
    ]
}
```

### With LangChain
```python
from langchain_pinecone import PineconeVectorStore

# Create
vectorstore = PineconeVectorStore.from_documents(
    documents=docs,
    embedding=embeddings,
    index_name="my-index",
    namespace="my-namespace"
)

# Load existing
vectorstore = PineconeVectorStore(
    index_name="my-index",
    embedding=embeddings,
    namespace="my-namespace"
)
```

### Pros & Cons
```
✓ Fully managed (zero ops)
✓ Auto-scaling
✓ Metadata filtering
✓ Namespaces for multi-tenancy
✓ High availability

✗ Vendor lock-in
✗ Cost at scale
✗ Data leaves your infrastructure
✗ Cold start latency (serverless)
```

---

## Qdrant

### Overview
- **Type**: Self-hosted or cloud
- **Best for**: Production, hybrid search, filtering
- **Index**: HNSW with custom optimizations

### Installation
```bash
pip install qdrant-client

# Run server
docker run -p 6333:6333 qdrant/qdrant
```

### Basic Usage
```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Connect
client = QdrantClient(":memory:")  # In-memory
client = QdrantClient(path="./qdrant_data")  # Local file
client = QdrantClient(host="localhost", port=6333)  # Server
client = QdrantClient(url="https://xxx.qdrant.io", api_key="xxxxx")  # Cloud

# Create collection
client.create_collection(
    collection_name="my_collection",
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE  # COSINE, EUCLID, DOT
    )
)

# Upsert points
client.upsert(
    collection_name="my_collection",
    points=[
        PointStruct(
            id=1,
            vector=[0.1, 0.2, ...],
            payload={"city": "London", "year": 2020}
        ),
        PointStruct(
            id=2,
            vector=[0.3, 0.4, ...],
            payload={"city": "Paris", "year": 2021}
        )
    ]
)

# Search
results = client.search(
    collection_name="my_collection",
    query_vector=[0.1, 0.2, ...],
    limit=5,
    query_filter={
        "must": [
            {"key": "city", "match": {"value": "London"}}
        ]
    }
)

# Search with score threshold
results = client.search(
    collection_name="my_collection",
    query_vector=[0.1, 0.2, ...],
    limit=5,
    score_threshold=0.8
)
```

### Advanced Filtering
```python
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

# Complex filter
filter = Filter(
    must=[
        FieldCondition(key="city", match=MatchValue(value="London")),
        FieldCondition(key="year", range=Range(gte=2020, lte=2022))
    ],
    should=[
        FieldCondition(key="category", match=MatchValue(value="tech")),
        FieldCondition(key="category", match=MatchValue(value="science"))
    ],
    must_not=[
        FieldCondition(key="deleted", match=MatchValue(value=True))
    ]
)

results = client.search(
    collection_name="my_collection",
    query_vector=vector,
    query_filter=filter,
    limit=10
)
```

### With LangChain
```python
from langchain_qdrant import QdrantVectorStore

# Create
vectorstore = QdrantVectorStore.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="my_collection",
    url="http://localhost:6333"
)

# Load existing
vectorstore = QdrantVectorStore(
    client=client,
    collection_name="my_collection",
    embedding=embeddings
)
```

### Pros & Cons
```
✓ Fast and efficient
✓ Rich filtering
✓ Payload (metadata) indexes
✓ Hybrid search support
✓ On-premise or cloud
✓ Rust-based (performance)

✗ Smaller community than others
✗ Self-hosting complexity
```

---

## Weaviate

### Overview
- **Type**: Self-hosted or cloud
- **Best for**: Hybrid search, GraphQL, multi-modal
- **Index**: HNSW with custom modules

### Installation
```bash
pip install weaviate-client

# Run server
docker run -p 8080:8080 semitechnologies/weaviate
```

### Basic Usage
```python
import weaviate

# Connect
client = weaviate.Client("http://localhost:8080")

# Cloud
client = weaviate.Client(
    url="https://xxx.weaviate.network",
    auth_client_secret=weaviate.AuthApiKey(api_key="xxxxx")
)

# Create schema
class_obj = {
    "class": "Article",
    "vectorizer": "text2vec-openai",  # Auto-embed
    "properties": [
        {"name": "title", "dataType": ["text"]},
        {"name": "content", "dataType": ["text"]},
        {"name": "category", "dataType": ["text"]}
    ]
}
client.schema.create_class(class_obj)

# Add objects
client.data_object.create(
    class_name="Article",
    data_object={
        "title": "My Article",
        "content": "Article content here",
        "category": "tech"
    }
)

# Add with vector
client.data_object.create(
    class_name="Article",
    data_object={"title": "My Article", "content": "..."},
    vector=[0.1, 0.2, ...]
)

# Search
result = client.query.get(
    "Article", ["title", "content"]
).with_near_text({
    "concepts": ["machine learning"]
}).with_limit(5).do()

# Hybrid search
result = client.query.get(
    "Article", ["title", "content"]
).with_hybrid(
    query="machine learning",
    alpha=0.5  # 0=keyword, 1=vector
).with_limit(5).do()

# Filter
result = client.query.get(
    "Article", ["title", "content"]
).with_near_text({
    "concepts": ["AI"]
}).with_where({
    "path": ["category"],
    "operator": "Equal",
    "valueText": "tech"
}).with_limit(5).do()
```

### With LangChain
```python
from langchain_weaviate import WeaviateVectorStore

vectorstore = WeaviateVectorStore.from_documents(
    documents=docs,
    embedding=embeddings,
    client=client,
    index_name="Article"
)
```

### Pros & Cons
```
✓ Built-in vectorizers (auto-embed)
✓ Native hybrid search
✓ GraphQL API
✓ Multi-modal (images, etc.)
✓ Modules ecosystem

✗ Complex setup
✗ Learning curve
✗ Resource intensive
```

---

## pgvector (PostgreSQL)

### Overview
- **Type**: PostgreSQL extension
- **Best for**: Existing Postgres users, simple use cases

### Installation
```bash
# PostgreSQL extension
CREATE EXTENSION vector;

# Python
pip install pgvector psycopg2-binary
```

### Basic Usage
```sql
-- Create table with vector column
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(384)
);

-- Insert
INSERT INTO documents (content, embedding)
VALUES ('Hello world', '[0.1, 0.2, ...]');

-- Search (L2 distance)
SELECT * FROM documents
ORDER BY embedding <-> '[0.1, 0.2, ...]'
LIMIT 5;

-- Cosine distance
SELECT * FROM documents
ORDER BY embedding <=> '[0.1, 0.2, ...]'
LIMIT 5;

-- Inner product
SELECT * FROM documents
ORDER BY embedding <#> '[0.1, 0.2, ...]'
LIMIT 5;

-- Create index (IVFFlat)
CREATE INDEX ON documents
USING ivfflat (embedding vector_l2_ops)
WITH (lists = 100);

-- Create index (HNSW)
CREATE INDEX ON documents
USING hnsw (embedding vector_l2_ops)
WITH (m = 16, ef_construction = 64);
```

### With Python
```python
import psycopg2
from pgvector.psycopg2 import register_vector

conn = psycopg2.connect("postgresql://localhost/mydb")
register_vector(conn)

cur = conn.cursor()

# Insert
cur.execute(
    "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
    ("Hello world", [0.1, 0.2, ...])
)

# Search
cur.execute(
    "SELECT * FROM documents ORDER BY embedding <-> %s LIMIT 5",
    ([0.1, 0.2, ...],)
)
results = cur.fetchall()
```

### With LangChain
```python
from langchain_postgres import PGVector

vectorstore = PGVector.from_documents(
    documents=docs,
    embedding=embeddings,
    connection="postgresql://user:pass@localhost/db",
    collection_name="my_docs"
)
```

### Pros & Cons
```
✓ Use existing Postgres
✓ ACID transactions
✓ SQL filtering
✓ No new infrastructure

✗ Not as fast as specialized DBs
✗ Limited scaling
✗ Fewer index options
```

---

## Choosing the Right Vector Database

### Decision Tree

```
Start
  │
  ├─ Prototype/Learning? ──────────► ChromaDB
  │
  ├─ Need managed service? ────────► Pinecone
  │
  ├─ Have PostgreSQL? ─────────────► pgvector
  │
  ├─ Need hybrid search? ──────────► Weaviate or Qdrant
  │
  ├─ Maximum performance? ─────────► FAISS (if no filtering)
  │                                  Qdrant (with filtering)
  │
  └─ Enterprise scale? ────────────► Milvus or Pinecone
```

### Comparison by Use Case

| Use Case | Best Choice | Reason |
|----------|-------------|--------|
| **Learning/Prototype** | ChromaDB | Zero setup, embedded |
| **Production SaaS** | Pinecone | Managed, scalable |
| **On-premise required** | Qdrant, Weaviate | Self-hosted options |
| **Existing Postgres** | pgvector | No new infrastructure |
| **Research/Custom** | FAISS | Maximum flexibility |
| **Hybrid search critical** | Weaviate | Native BM25 + vector |
| **Cost-sensitive** | Qdrant Cloud | Generous free tier |

### Performance Comparison (Approximate)

| Database | QPS (1M vectors) | Latency p99 |
|----------|------------------|-------------|
| FAISS (HNSW) | 3000+ | < 5ms |
| Qdrant | 1500+ | < 10ms |
| Weaviate | 1000+ | < 15ms |
| Pinecone | 1000+ | < 20ms |
| ChromaDB | 500+ | < 30ms |
| pgvector | 300+ | < 50ms |

*Note: Numbers vary significantly based on hardware, index config, and data*

---

## Interview Questions

**Q: When would you choose FAISS over Pinecone?**
> FAISS when: need maximum speed, no metadata filtering, have ML expertise, cost-sensitive, on-premise required.
> Pinecone when: need managed service, want zero-ops, need filtering, have budget.

**Q: How does HNSW work?**
> HNSW builds a multi-layer graph where each layer has fewer nodes. Search starts from top layer, greedily navigates to closest nodes, then moves down layers for finer search. O(log n) search complexity.

**Q: How do you handle vector database updates?**
> Options: 1) Upsert (replace), 2) Delete + Insert, 3) Soft delete with filtering. FAISS requires index rebuild; others support incremental updates.

**Q: How do you scale vector search?**
> 1) Sharding (split by ID/metadata), 2) Replicas (read scaling), 3) Quantization (reduce memory), 4) Approximate indexes (trade accuracy for speed).
