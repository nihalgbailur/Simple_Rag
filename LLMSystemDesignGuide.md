# LLM System Design Guide

How to design production-ready LLM applications - architecture patterns, scaling, and real-world considerations.

---

## System Design Framework

When asked to design an LLM system, follow this framework:

```
1. Clarify Requirements
   ├── Functional: What should it do?
   ├── Non-functional: Scale, latency, cost?
   └── Constraints: Budget, timeline, team?

2. High-Level Design
   ├── Core components
   ├── Data flow
   └── Technology choices

3. Deep Dive
   ├── Critical components
   ├── Failure handling
   └── Scaling strategy

4. Trade-offs
   ├── Cost vs quality
   ├── Latency vs accuracy
   └── Build vs buy
```

---

## Common LLM System Architectures

### 1. Simple Chatbot

```
┌─────────────────────────────────────────────────────────┐
│                     Simple Chatbot                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   User ──► API Gateway ──► LLM Service ──► Response     │
│                │                                         │
│                ▼                                         │
│         Rate Limiter                                     │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Components:**
- API Gateway: Handle requests, authentication
- Rate Limiter: Prevent abuse
- LLM Service: Call OpenAI/Claude/etc.

---

### 2. RAG System

```
┌──────────────────────────────────────────────────────────────────┐
│                         RAG System                                │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │  Documents  │───►│  Chunking   │───►│  Embedding  │          │
│  └─────────────┘    └─────────────┘    └──────┬──────┘          │
│                                                │                  │
│                                                ▼                  │
│                                        ┌─────────────┐           │
│                                        │ Vector DB   │           │
│                                        └──────┬──────┘           │
│                                               │                   │
│  ┌─────────────┐    ┌─────────────┐          │                   │
│  │    User     │───►│   Query     │──────────┤                   │
│  └─────────────┘    └──────┬──────┘          │                   │
│                            │                  ▼                   │
│                            │         ┌─────────────┐             │
│                            │         │  Retriever  │             │
│                            │         └──────┬──────┘             │
│                            │                │                     │
│                            ▼                ▼                     │
│                     ┌─────────────────────────────┐              │
│                     │     Prompt Construction     │              │
│                     │   (Query + Context)         │              │
│                     └─────────────┬───────────────┘              │
│                                   │                               │
│                                   ▼                               │
│                           ┌─────────────┐                        │
│                           │     LLM     │                        │
│                           └──────┬──────┘                        │
│                                  │                                │
│                                  ▼                                │
│                           ┌─────────────┐                        │
│                           │  Response   │                        │
│                           └─────────────┘                        │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

### 3. Agentic System

```
┌──────────────────────────────────────────────────────────────────┐
│                       Agentic System                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│   User Query                                                      │
│       │                                                           │
│       ▼                                                           │
│   ┌─────────────┐                                                │
│   │   Planner   │ ◄─── LLM decides what to do                    │
│   └──────┬──────┘                                                │
│          │                                                        │
│          ▼                                                        │
│   ┌─────────────────────────────────────────┐                    │
│   │            Tool Selection                │                    │
│   ├─────────┬─────────┬─────────┬──────────┤                    │
│   │ Search  │   SQL   │  Code   │   API    │                    │
│   │  Tool   │  Tool   │  Tool   │   Tool   │                    │
│   └────┬────┴────┬────┴────┬────┴────┬─────┘                    │
│        │         │         │         │                           │
│        └─────────┴────┬────┴─────────┘                           │
│                       │                                           │
│                       ▼                                           │
│               ┌─────────────┐                                    │
│               │   Execute   │                                    │
│               └──────┬──────┘                                    │
│                      │                                            │
│                      ▼                                            │
│               ┌─────────────┐                                    │
│               │  Observe    │ ──► Need more? ──► Loop back       │
│               └──────┬──────┘                                    │
│                      │                                            │
│                      ▼                                            │
│               ┌─────────────┐                                    │
│               │   Answer    │                                    │
│               └─────────────┘                                    │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

### 4. Multi-Model Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    Multi-Model Architecture                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│   User Query                                                      │
│       │                                                           │
│       ▼                                                           │
│   ┌─────────────┐                                                │
│   │   Router    │ ◄─── Classify query complexity                 │
│   └──────┬──────┘                                                │
│          │                                                        │
│   ┌──────┴──────┬──────────────┐                                 │
│   │             │              │                                  │
│   ▼             ▼              ▼                                  │
│ Simple       Medium        Complex                                │
│   │             │              │                                  │
│   ▼             ▼              ▼                                  │
│ ┌─────┐    ┌─────────┐   ┌─────────┐                            │
│ │GPT-3│    │GPT-3.5  │   │  GPT-4  │                            │
│ │Mini │    │ Turbo   │   │         │                            │
│ └──┬──┘    └────┬────┘   └────┬────┘                            │
│    │            │             │                                   │
│    └────────────┴─────────────┘                                  │
│                 │                                                 │
│                 ▼                                                 │
│            Response                                               │
│                                                                   │
│   Cost Savings: 50-70% by routing simple queries to smaller LLMs │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Design Example: Customer Support Chatbot

### Requirements Clarification

```
Functional:
- Answer customer questions about products
- Handle returns and refunds
- Escalate to human when needed
- Support multiple languages

Non-Functional:
- 10K concurrent users
- < 3 second response time
- 99.9% uptime
- Cost under $10K/month
```

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Customer Support Chatbot                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │  Client  │───►│ Load Balancer│───►│  API Gateway │                  │
│  │  (Web)   │    │              │    │  + Auth      │                  │
│  └──────────┘    └──────────────┘    └───────┬──────┘                  │
│                                              │                           │
│                         ┌────────────────────┼────────────────────┐     │
│                         │                    │                    │     │
│                         ▼                    ▼                    ▼     │
│                   ┌──────────┐        ┌──────────┐        ┌──────────┐ │
│                   │  Intent  │        │ Session  │        │   Rate   │ │
│                   │Classifier│        │  Store   │        │ Limiter  │ │
│                   └────┬─────┘        │ (Redis)  │        └──────────┘ │
│                        │              └──────────┘                      │
│          ┌─────────────┼─────────────┐                                 │
│          │             │             │                                  │
│          ▼             ▼             ▼                                  │
│    ┌──────────┐  ┌──────────┐  ┌──────────┐                           │
│    │   FAQ    │  │  Action  │  │ Escalate │                           │
│    │   RAG    │  │  Agent   │  │  Human   │                           │
│    └────┬─────┘  └────┬─────┘  └────┬─────┘                           │
│         │             │             │                                   │
│         │             ▼             │                                   │
│         │      ┌──────────────┐    │                                   │
│         │      │   Tools:     │    │                                   │
│         │      │ - Order API  │    │                                   │
│         │      │ - Refund API │    │                                   │
│         │      │ - Product DB │    │                                   │
│         │      └──────────────┘    │                                   │
│         │             │             │                                   │
│         └─────────────┴─────────────┘                                  │
│                       │                                                 │
│                       ▼                                                 │
│               ┌──────────────┐      ┌──────────────┐                   │
│               │   Response   │─────►│   Logging    │                   │
│               │  Generator   │      │  + Metrics   │                   │
│               └──────────────┘      └──────────────┘                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Deep Dive

**1. Intent Classifier**
```python
intents = {
    "faq": "General questions → RAG",
    "order_status": "Check order → Action Agent",
    "refund": "Process refund → Action Agent",
    "complaint": "Angry customer → Escalate",
    "unknown": "Can't classify → Escalate"
}

# Use small classifier model or LLM
def classify_intent(query):
    # Option 1: Fine-tuned BERT classifier (fast, cheap)
    # Option 2: Few-shot LLM classification (flexible)
    return intent, confidence
```

**2. FAQ RAG System**
```
- Vector DB: Pinecone (managed, scalable)
- Embedding: text-embedding-3-small (cheap)
- Chunks: 500 tokens with 50 overlap
- Retrieval: Top 3 chunks
- LLM: GPT-3.5-turbo (cost-effective)
```

**3. Action Agent**
```python
tools = [
    Tool("check_order", "Get order status by order_id"),
    Tool("process_refund", "Initiate refund for order_id"),
    Tool("get_product", "Get product details by product_id"),
]

# GPT-4 for complex reasoning
# Function calling for structured tool use
```

**4. Session Store (Redis)**
```python
session = {
    "user_id": "123",
    "conversation_history": [...],  # Last 10 messages
    "context": {...},  # Order info, user preferences
    "created_at": "...",
    "ttl": 3600  # 1 hour
}
```

### Scaling Strategy

```
Traffic Tiers:
┌─────────────┬──────────────────────────────────────┐
│   Users     │   Architecture                       │
├─────────────┼──────────────────────────────────────┤
│   < 100     │   Single server + API calls          │
│   100-1K    │   + Redis, Async processing          │
│   1K-10K    │   + Load balancer, Multiple workers  │
│   10K-100K  │   + Queue (SQS), Auto-scaling        │
│   > 100K    │   + Regional deployment, CDN         │
└─────────────┴──────────────────────────────────────┘
```

### Cost Estimation

```
Assumptions: 10K users, 5 queries/user/day = 50K queries/day

Costs:
┌────────────────────┬─────────────────┬───────────────┐
│ Component          │ Per Query       │ Monthly       │
├────────────────────┼─────────────────┼───────────────┤
│ Embedding          │ $0.0001         │ $150          │
│ GPT-3.5 (FAQ)      │ $0.002          │ $2,000        │
│ GPT-4 (Actions)    │ $0.03 (20%)     │ $9,000        │
│ Vector DB          │ -               │ $70           │
│ Infrastructure     │ -               │ $500          │
├────────────────────┼─────────────────┼───────────────┤
│ Total              │                 │ ~$11,720      │
└────────────────────┴─────────────────┴───────────────┘

Optimizations to hit $10K:
- Cache frequent queries (30% savings)
- Use smaller model for simple queries
- Batch embeddings
```

---

## Design Example: Document Q&A System

### Requirements

```
- Upload documents (PDF, Word, TXT)
- Ask questions about documents
- Support 1000 documents per user
- Multi-user with isolation
- Source citations in answers
```

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Document Q&A System                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Upload Flow:                                                            │
│  ┌────────┐   ┌────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │ Upload │──►│  S3    │──►│  Queue   │──►│ Worker   │──►│Vector DB │ │
│  │  API   │   │        │   │ (SQS)    │   │(Process) │   │(Per User)│ │
│  └────────┘   └────────┘   └──────────┘   └──────────┘   └──────────┘ │
│                                                │                        │
│                                                ▼                        │
│                                         ┌──────────┐                   │
│                                         │ Metadata │                   │
│                                         │   DB     │                   │
│                                         │(Postgres)│                   │
│                                         └──────────┘                   │
│                                                                         │
│  Query Flow:                                                            │
│  ┌────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐             │
│  │ Query  │──►│ Retrieve │──►│  Build   │──►│   LLM    │             │
│  │  API   │   │  Chunks  │   │  Prompt  │   │ Generate │             │
│  └────────┘   └──────────┘   └──────────┘   └────┬─────┘             │
│                     │                             │                     │
│                     ▼                             ▼                     │
│              ┌──────────┐                  ┌──────────┐                │
│              │ User's   │                  │ Response │                │
│              │Vector DB │                  │ + Sources│                │
│              └──────────┘                  └──────────┘                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Multi-Tenancy Strategy

```
Option 1: Namespace per user (Pinecone)
- Same index, different namespaces
- Easy but limited isolation

Option 2: Collection per user (Chroma/Qdrant)
- Separate collections
- Better isolation, more overhead

Option 3: Metadata filtering
- All in one, filter by user_id
- Simplest but security risk

Recommendation: Collection per user for sensitive data
```

---

## Key Design Patterns

### 1. Caching Layer

```
┌─────────────────────────────────────────────────┐
│               Caching Strategy                   │
├─────────────────────────────────────────────────┤
│                                                  │
│  Query ──► Hash(query) ──► Check Cache          │
│                               │                  │
│                    ┌──────────┴──────────┐      │
│                    │                     │      │
│                  Cache Hit           Cache Miss  │
│                    │                     │      │
│                    ▼                     ▼      │
│              Return Cached         Call LLM     │
│                                         │       │
│                                         ▼       │
│                                   Store in Cache│
│                                         │       │
│                                         ▼       │
│                                    Return       │
│                                                  │
│  Cache Types:                                   │
│  - Exact match: Same query = same response     │
│  - Semantic: Similar queries = cached response │
│                                                  │
└─────────────────────────────────────────────────┘
```

### 2. Fallback Chain

```python
def query_with_fallback(query):
    try:
        # Try primary (GPT-4)
        return call_gpt4(query)
    except RateLimitError:
        # Fallback to secondary
        return call_gpt35(query)
    except ServiceError:
        # Fallback to tertiary
        return call_claude(query)
    except:
        # Final fallback
        return "I'm experiencing issues. Please try again."
```

### 3. Request Queue

```
┌─────────────────────────────────────────────────┐
│             Async Processing                     │
├─────────────────────────────────────────────────┤
│                                                  │
│   Request ──► Queue ──► Workers ──► Callback    │
│                │                                 │
│                ▼                                 │
│   Benefits:                                      │
│   - Handle traffic spikes                       │
│   - Retry failed requests                       │
│   - Rate limit management                       │
│   - Cost smoothing                              │
│                                                  │
│   Implementation:                               │
│   - AWS SQS + Lambda                           │
│   - Redis Queue + Workers                      │
│   - Celery + RabbitMQ                          │
│                                                  │
└─────────────────────────────────────────────────┘
```

### 4. Streaming Architecture

```
┌─────────────────────────────────────────────────┐
│              Streaming Response                  │
├─────────────────────────────────────────────────┤
│                                                  │
│   Client ◄──── SSE/WebSocket ◄──── Server       │
│                                      │          │
│                              ┌───────┴───────┐  │
│                              │ LLM Streaming │  │
│                              │    API        │  │
│                              └───────────────┘  │
│                                                  │
│   Benefits:                                      │
│   - Lower perceived latency                     │
│   - Better UX (see tokens appear)              │
│   - Can cancel mid-response                    │
│                                                  │
│   Implementation:                               │
│   - Server-Sent Events (SSE)                   │
│   - WebSocket                                  │
│   - HTTP/2 streaming                           │
│                                                  │
└─────────────────────────────────────────────────┘
```

---

## Reliability Patterns

### Circuit Breaker

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failures = 0
        self.threshold = failure_threshold
        self.timeout = timeout
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.last_failure = None

    def call(self, func):
        if self.state == "OPEN":
            if time.time() - self.last_failure > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise CircuitOpenError()

        try:
            result = func()
            self.failures = 0
            self.state = "CLOSED"
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure = time.time()
            if self.failures >= self.threshold:
                self.state = "OPEN"
            raise e
```

### Retry with Exponential Backoff

```python
import time
import random

def retry_with_backoff(func, max_retries=3, base_delay=1):
    for attempt in range(max_retries):
        try:
            return func()
        except (RateLimitError, TimeoutError) as e:
            if attempt == max_retries - 1:
                raise e
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            time.sleep(delay)
```

---

## Interview Tips

### Common Questions

**Q: Design a chatbot for an e-commerce site**
```
Key points:
1. Intent classification (browse, buy, support)
2. Product search (RAG over catalog)
3. Order management (tool use)
4. Personalization (user history)
5. Handoff to human
```

**Q: How do you handle LLM failures?**
```
1. Retry with backoff
2. Fallback to simpler model
3. Fallback to cached response
4. Graceful degradation message
5. Circuit breaker for repeated failures
```

**Q: How do you reduce costs?**
```
1. Cache responses (exact + semantic)
2. Route to cheaper models when possible
3. Prompt optimization (shorter prompts)
4. Batch requests where possible
5. Use embeddings cache
6. Set token limits
```

**Q: How do you ensure low latency?**
```
1. Streaming responses
2. Edge caching
3. Async processing
4. Model routing (fast for simple)
5. Precompute common queries
6. Optimize retrieval (ANN tuning)
```

**Q: How do you handle scale?**
```
1. Horizontal scaling (stateless services)
2. Queue-based processing
3. Auto-scaling based on queue depth
4. Regional deployment
5. Rate limiting per user
6. Caching at multiple levels
```
