# RAG Chatbot - Interview Guide

A simple Retrieval-Augmented Generation (RAG) chatbot built with LangChain, Ollama, and ChromaDB.

---

## How RAG Works (Interview Explanation)

### The Problem RAG Solves
LLMs have two major limitations:
1. **Knowledge cutoff** - They don't know about recent events or your private data
2. **Hallucinations** - They sometimes make up information confidently

RAG solves both by **retrieving relevant context** from your documents before generating answers.

### RAG Pipeline (Step-by-Step)

```
┌─────────────────────────────────────────────────────────────────┐
│                      INDEXING PHASE (One-time)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   PDF Documents                                                 │
│        │                                                        │
│        ▼                                                        │
│   ┌─────────┐    "Split large documents into                    │
│   │ CHUNKING │    smaller pieces (500 chars)                    │
│   └────┬────┘    for better retrieval precision"                │
│        │                                                        │
│        ▼                                                        │
│   ┌──────────┐   "Convert text to vectors                       │
│   │EMBEDDING │   (numerical representations)                    │
│   └────┬─────┘   using sentence-transformers"                   │
│        │                                                        │
│        ▼                                                        │
│   ┌───────────┐  "Store vectors in ChromaDB                     │
│   │VECTOR STORE│  for fast similarity search"                   │
│   └───────────┘                                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    QUERY PHASE (Every question)                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   User Question: "What is the main topic?"                      │
│        │                                                        │
│        ▼                                                        │
│   ┌──────────┐   "Convert question to vector                    │
│   │EMBEDDING │    using same model"                             │
│   └────┬─────┘                                                  │
│        │                                                        │
│        ▼                                                        │
│   ┌──────────┐   "Find top-k most similar                       │
│   │RETRIEVAL │    chunks using cosine similarity"               │
│   └────┬─────┘                                                  │
│        │                                                        │
│        ▼                                                        │
│   ┌──────────┐   "Combine: Prompt + Context + Question          │
│   │   LLM    │    and generate grounded answer"                 │
│   └────┬─────┘                                                  │
│        │                                                        │
│        ▼                                                        │
│   Answer with source citations                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Components Explained

### 1. Document Loading
```python
loader = DirectoryLoader(DOCUMENTS_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
```
- Loads all PDFs from a folder
- Extracts text from each page
- Preserves metadata (source file, page number)

### 2. Text Chunking
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
```
- **Why chunk?** Large documents don't fit in LLM context, and retrieval is more precise with smaller chunks
- **chunk_size=500**: Each chunk is ~500 characters
- **chunk_overlap=50**: Overlapping prevents cutting sentences in the middle

### 3. Embeddings
```python
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```
- Converts text to 384-dimensional vectors
- Similar meanings → similar vectors
- Runs locally, completely free

### 4. Vector Store (ChromaDB)
```python
vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
```
- Stores vectors with their text
- Enables fast similarity search
- Persists to disk for reuse

### 5. Retrieval
```python
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
```
- Finds top 3 most relevant chunks
- Uses cosine similarity between query and stored vectors

### 6. Generation (Ollama LLM)
```python
llm = Ollama(model="phi3:mini")
```
- Local LLM, no API costs
- Generates answer using retrieved context
- Prompt template ensures grounded responses

---

## Interview Questions & Answers

### Basic Questions

**Q1: What is RAG?**
> RAG (Retrieval-Augmented Generation) is a technique that enhances LLM responses by first retrieving relevant information from a knowledge base, then using that context to generate more accurate, grounded answers. It reduces hallucinations and allows LLMs to access private or recent data.

**Q2: Why do we need chunking?**
> 1. LLMs have limited context windows
> 2. Smaller chunks improve retrieval precision
> 3. Large documents would dilute the relevance signal
> 4. Overlap prevents losing context at chunk boundaries

**Q3: What are embeddings?**
> Embeddings are dense vector representations of text where semantically similar content has similar vectors. They capture meaning, not just keywords. For example, "car" and "automobile" would have similar embeddings even though they share no letters.

**Q4: Why use vector databases instead of keyword search?**
> Vector search finds semantically similar content, not just exact matches. If someone asks "What's the revenue?" it can find chunks mentioning "income", "earnings", or "sales figures" even without the word "revenue".

**Q5: What is cosine similarity?**
> Cosine similarity measures the angle between two vectors. Value ranges from -1 to 1, where 1 means identical direction (most similar). It's preferred over Euclidean distance because it's magnitude-independent.

### Intermediate Questions

**Q6: How do you choose chunk size?**
> It's a tradeoff:
> - **Too small**: Loses context, incomplete information
> - **Too large**: Retrieves irrelevant content, less precise
> - **Typical range**: 256-1024 characters
> - Best practice: Experiment with your specific data

**Q7: What is chunk overlap and why use it?**
> Overlap ensures sentences aren't cut in half between chunks. If a key sentence spans two chunks, the overlap ensures it's fully captured in at least one chunk. Typically 10-20% of chunk size.

**Q8: How many chunks should you retrieve (k)?**
> Depends on:
> - LLM context window size
> - Chunk size
> - Question complexity
> - Typical: 3-5 chunks
> - More isn't always better (can add noise)

**Q9: What's the difference between stuff, map_reduce, and refine chain types?**
> - **Stuff**: Puts all chunks in one prompt (simple, limited by context)
> - **Map_reduce**: Processes each chunk separately, then combines (parallel, good for summarization)
> - **Refine**: Iteratively refines answer with each chunk (sequential, maintains coherence)

**Q10: How do you handle multiple document types?**
> Use appropriate loaders:
> - PDFs: PyPDFLoader
> - Word docs: Docx2txtLoader
> - Web pages: WebBaseLoader
> - CSVs: CSVLoader
> LangChain provides loaders for most formats.

### Advanced Questions

**Q11: What are the limitations of basic RAG?**
> 1. **Lost in the middle**: LLMs may ignore middle chunks
> 2. **No multi-hop reasoning**: Can't connect info across chunks
> 3. **Retrieval failures**: Wrong chunks = wrong answers
> 4. **No conversation memory**: Each query is independent

**Q12: What is Hybrid Search?**
> Combines vector search (semantic) with keyword search (BM25):
> - Vector: Good for meaning, bad for exact terms
> - BM25: Good for specific keywords, names, codes
> - Hybrid: Best of both, usually weighted combination

**Q13: What is Re-ranking?**
> After initial retrieval, a cross-encoder model re-scores each chunk against the query for more accurate relevance. Two-stage process:
> 1. Fast retrieval: Get top 20 chunks (bi-encoder)
> 2. Accurate re-rank: Score and keep top 3 (cross-encoder)

**Q14: What is HyDE (Hypothetical Document Embedding)?**
> Instead of embedding the question directly:
> 1. Ask LLM to generate a hypothetical answer
> 2. Embed that hypothetical answer
> 3. Search with this embedding
> Works because the hypothetical answer is closer to actual document language.

**Q15: How would you evaluate a RAG system?**
> Metrics:
> - **Retrieval**: Precision@k, Recall@k, MRR (Mean Reciprocal Rank)
> - **Generation**: Faithfulness, Answer Relevance, Groundedness
> - **End-to-end**: Human evaluation, RAGAS framework
>
> Key test: Does the answer come from retrieved context?

**Q16: How do you handle hallucinations in RAG?**
> 1. Prompt engineering: "Only use provided context"
> 2. Lower temperature (0.0-0.3)
> 3. Add "I don't know" instructions
> 4. Citation requirements
> 5. Fact-checking layer

**Q17: What's the difference between RAG and fine-tuning?**
> | Aspect | RAG | Fine-tuning |
> |--------|-----|-------------|
> | Knowledge | External, updatable | Baked into weights |
> | Cost | Cheaper, no training | Expensive training |
> | Updates | Easy (update docs) | Retrain needed |
> | Use case | Dynamic knowledge | Behavior change |

---

## Quick Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start Ollama
ollama serve

# 4. Add PDFs to documents/ folder

# 5. Run
python app.py
```

---

## Tech Stack Summary

| Component | Tool | Why |
|-----------|------|-----|
| Framework | LangChain | Industry standard, great abstractions |
| Vector DB | ChromaDB | Simple, embedded, no setup |
| Embeddings | HuggingFace | Free, local, good quality |
| LLM | Ollama | Local, private, no API costs |
| PDF Loading | PyPDF | Reliable PDF text extraction |

---

## Code Structure

```
app.py
├── load_documents()      # Load PDFs
├── split_documents()     # Chunk text
├── create_vector_store() # Embed & store
├── load_vector_store()   # Load existing DB
├── create_rag_chain()    # Build QA pipeline
└── main()                # Chat interface
```
