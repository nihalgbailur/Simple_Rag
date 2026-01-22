# Production MLOps for LLMs Guide

How to deploy, monitor, and maintain LLM applications in production.

---

## LLMOps Overview

```
Traditional MLOps:
Train → Deploy → Monitor → Retrain

LLMOps:
Prompt Engineering → Deploy → Monitor → Iterate Prompts
       └── Fine-tune (optional) ──────┘
```

---

## Deployment Architectures

### 1. API-Based (Most Common)

```
┌─────────────────────────────────────────────────────────┐
│                  API-Based Architecture                  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Your App ──► Your API ──► OpenAI/Anthropic API        │
│                  │                                       │
│                  ├── Rate limiting                      │
│                  ├── Caching                            │
│                  ├── Logging                            │
│                  └── Fallbacks                          │
│                                                          │
│  Pros: No infrastructure, always latest models          │
│  Cons: Vendor dependency, data privacy, costs at scale  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 2. Self-Hosted LLMs

```
┌─────────────────────────────────────────────────────────┐
│               Self-Hosted Architecture                   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Your App ──► Load Balancer ──► LLM Inference Servers  │
│                                      │                   │
│                              ┌───────┴───────┐          │
│                              │               │          │
│                           GPU Server 1   GPU Server 2   │
│                           (vLLM/TGI)     (vLLM/TGI)     │
│                                                          │
│  Pros: Data privacy, cost control, customization        │
│  Cons: Infrastructure complexity, GPU costs             │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Self-Hosting LLMs

### vLLM (Recommended)

```bash
pip install vllm

python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --port 8000
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Ollama (Easiest)

```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3
ollama serve
```

### GPU Requirements

| Model Size | Min VRAM | Recommended |
|------------|----------|-------------|
| 7B | 8 GB | 16 GB |
| 13B | 16 GB | 24 GB |
| 70B | 80 GB | 2x 80 GB |

---

## Containerization

### Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - redis
      - chromadb
  redis:
    image: redis:7-alpine
  chromadb:
    image: chromadb/chroma:latest
```

---

## Observability

### Key Metrics

```python
from prometheus_client import Counter, Histogram

llm_requests_total = Counter('llm_requests_total', 'Total requests', ['model', 'status'])
llm_latency_seconds = Histogram('llm_latency_seconds', 'Latency', ['model'])
llm_tokens_total = Counter('llm_tokens_total', 'Tokens', ['model', 'type'])
llm_cost_dollars = Counter('llm_cost_dollars', 'Cost', ['model'])
```

### Logging

```python
import logging
import json

class LLMLogger:
    def log_request(self, request_id, model, prompt_length):
        logging.info(json.dumps({
            "type": "llm_request",
            "request_id": request_id,
            "model": model,
            "prompt_length": prompt_length
        }))

    def log_response(self, request_id, latency_ms, tokens):
        logging.info(json.dumps({
            "type": "llm_response",
            "request_id": request_id,
            "latency_ms": latency_ms,
            "tokens": tokens
        }))
```

### LangSmith Tracing

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-key"
os.environ["LANGCHAIN_PROJECT"] = "production"
```

---

## Cost Management

### Token Pricing

```python
PRICING = {
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
}

def calculate_cost(model, input_tokens, output_tokens):
    pricing = PRICING.get(model)
    return (input_tokens/1000) * pricing["input"] + (output_tokens/1000) * pricing["output"]
```

### Cost Optimization

```python
# 1. Caching
class LLMCache:
    def __init__(self, redis_client):
        self.redis = redis_client

    def get(self, prompt, model):
        key = hashlib.md5(f"{model}:{prompt}".encode()).hexdigest()
        return self.redis.get(key)

# 2. Model routing
def route_to_model(complexity_score):
    if complexity_score < 0.3:
        return "gpt-3.5-turbo"
    elif complexity_score < 0.7:
        return "gpt-4-turbo"
    return "gpt-4"
```

---

## Security

### API Key Management

```python
# Environment variables
api_key = os.environ.get("OPENAI_API_KEY")

# AWS Secrets Manager
import boto3
def get_secret(name):
    client = boto3.client('secretsmanager')
    return client.get_secret_value(SecretId=name)['SecretString']
```

### Input Validation

```python
def sanitize_input(user_input: str) -> str:
    dangerous = ["ignore previous instructions", "system prompt"]
    for pattern in dangerous:
        if pattern.lower() in user_input.lower():
            raise ValueError("Invalid input")
    return user_input[:10000]  # Limit length
```

### Rate Limiting

```python
class RateLimiter:
    def __init__(self, rpm=60):
        self.rpm = rpm
        self.requests = {}

    def is_allowed(self, user_id):
        now = time.time()
        self.requests[user_id] = [t for t in self.requests.get(user_id, []) if t > now - 60]
        if len(self.requests.get(user_id, [])) >= self.rpm:
            return False
        self.requests.setdefault(user_id, []).append(now)
        return True
```

### PII Handling

```python
import re

def redact_pii(text):
    patterns = {
        r'\b[\w.]+@[\w.]+\.\w+\b': '[EMAIL]',
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b': '[PHONE]',
        r'\b\d{3}-\d{2}-\d{4}\b': '[SSN]',
    }
    for pattern, replacement in patterns.items():
        text = re.sub(pattern, replacement, text)
    return text
```

---

## Testing

### Unit Tests

```python
def test_rag_retrieval():
    mock_vectorstore = Mock()
    mock_vectorstore.similarity_search.return_value = [
        Document(page_content="Python is a language")
    ]
    retriever = RAGRetriever(vectorstore=mock_vectorstore)
    results = retriever.retrieve("What is Python?")
    assert len(results) > 0
```

### Load Testing

```python
async def load_test(url, num_requests=100):
    async with aiohttp.ClientSession() as session:
        tasks = [make_request(session, url) for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)
    latencies = [r["latency"] for r in results]
    print(f"P95: {sorted(latencies)[95]:.2f}s")
```

---

## CI/CD Pipeline

```yaml
name: Deploy
on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - run: docker build -t app:${{ github.sha }} .
      - run: kubectl set image deployment/app app=app:${{ github.sha }}
```

---

## Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-app
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: app
        image: rag-app:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: secrets
              key: openai-key
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        averageUtilization: 70
```

---

## Interview Questions

**Q: How do you handle LLM API failures?**
> Retry with backoff, circuit breaker, fallback models, caching, graceful degradation.

**Q: How do you reduce latency?**
> Streaming, caching, smaller models for simple queries, async processing.

**Q: How do you monitor LLM apps?**
> Track latency, error rate, tokens, cost, cache hits. Use LangSmith, Prometheus, Grafana.

**Q: How do you handle sensitive data?**
> Redact PII, self-host for sensitive data, encryption, access controls.
