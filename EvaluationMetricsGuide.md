# LLM & RAG Evaluation Guide

How to measure and assess the quality of LLM applications - metrics, frameworks, and best practices.

---

## Why Evaluation Matters

```
Without evaluation:
- No idea if changes improve or hurt quality
- Can't compare models or approaches
- No way to catch regressions
- Can't justify decisions to stakeholders

With evaluation:
- Data-driven improvements
- Confident deployments
- Clear quality benchmarks
- Measurable progress
```

---

## Evaluation Types

### 1. Offline Evaluation
- Run on test datasets before deployment
- Compare models/prompts/configs
- Automated, repeatable

### 2. Online Evaluation
- Measure in production
- Real user interactions
- A/B testing

### 3. Human Evaluation
- Expert review
- User feedback
- Gold standard for quality

---

## RAG Evaluation: The RAG Triad

```
┌─────────────────────────────────────────────────────────┐
│                    RAG TRIAD                             │
├─────────────────────────────────────────────────────────┤
│                                                          │
│                    Question                              │
│                       │                                  │
│            ┌──────────┴──────────┐                      │
│            │                     │                      │
│            ▼                     │                      │
│     ┌──────────────┐            │                      │
│     │  Retrieved   │            │                      │
│     │   Context    │            │                      │
│     └──────┬───────┘            │                      │
│            │                     │                      │
│     ┌──────┴───────┐            │                      │
│     │              │            │                      │
│     ▼              ▼            ▼                      │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐               │
│  │Context  │  │Grounded-│  │ Answer  │               │
│  │Relevance│  │  ness   │  │Relevance│               │
│  └─────────┘  └─────────┘  └─────────┘               │
│       │            │            │                      │
│       │            │            │                      │
│       ▼            ▼            ▼                      │
│   Is context    Is answer    Does answer              │
│   relevant to   based on     address the              │
│   question?     context?     question?                │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 1. Context Relevance
**Question:** Is the retrieved context relevant to the question?

```python
# What to measure:
- Are the right chunks retrieved?
- Is irrelevant information included?

# Metrics:
- Precision@k: % of retrieved docs that are relevant
- Recall@k: % of relevant docs that were retrieved
- MRR: Mean Reciprocal Rank
- NDCG: Normalized Discounted Cumulative Gain

# Failure mode:
- Wrong chunks → Wrong answer
```

### 2. Groundedness (Faithfulness)
**Question:** Is the answer supported by the retrieved context?

```python
# What to measure:
- Are claims in the answer traceable to context?
- Is the LLM hallucinating information?

# Metrics:
- Claim verification rate
- Source attribution accuracy
- Hallucination rate

# Failure mode:
- Good context but LLM makes things up
```

### 3. Answer Relevance
**Question:** Does the answer address the original question?

```python
# What to measure:
- Is the response on-topic?
- Does it fully answer the question?

# Metrics:
- Semantic similarity to expected answer
- Question-answer relevance score
- Completeness check

# Failure mode:
- Technically correct but doesn't answer what was asked
```

---

## Retrieval Metrics

### Precision@k

```python
def precision_at_k(retrieved_docs, relevant_docs, k):
    """What fraction of retrieved docs are relevant?"""
    retrieved_k = retrieved_docs[:k]
    relevant_retrieved = len(set(retrieved_k) & set(relevant_docs))
    return relevant_retrieved / k

# Example:
# Retrieved: [A, B, C, D, E] (k=5)
# Relevant: [A, C, F, G]
# Precision@5 = 2/5 = 0.4
```

### Recall@k

```python
def recall_at_k(retrieved_docs, relevant_docs, k):
    """What fraction of relevant docs were retrieved?"""
    retrieved_k = retrieved_docs[:k]
    relevant_retrieved = len(set(retrieved_k) & set(relevant_docs))
    return relevant_retrieved / len(relevant_docs)

# Example:
# Retrieved: [A, B, C, D, E] (k=5)
# Relevant: [A, C, F, G]
# Recall@5 = 2/4 = 0.5
```

### Mean Reciprocal Rank (MRR)

```python
def mrr(queries_results):
    """Average of 1/rank of first relevant result"""
    reciprocal_ranks = []
    for results, relevant in queries_results:
        for i, doc in enumerate(results, 1):
            if doc in relevant:
                reciprocal_ranks.append(1/i)
                break
        else:
            reciprocal_ranks.append(0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)

# Example:
# Query 1: First relevant at position 1 → 1/1 = 1.0
# Query 2: First relevant at position 3 → 1/3 = 0.33
# MRR = (1.0 + 0.33) / 2 = 0.67
```

### NDCG (Normalized Discounted Cumulative Gain)

```python
import numpy as np

def dcg_at_k(relevances, k):
    """Discounted Cumulative Gain"""
    relevances = np.array(relevances)[:k]
    gains = 2**relevances - 1
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    return np.sum(gains / discounts)

def ndcg_at_k(relevances, k):
    """Normalized DCG - accounts for position"""
    dcg = dcg_at_k(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)
    return dcg / idcg if idcg > 0 else 0

# Example:
# Relevances: [3, 2, 0, 1, 0] (higher = more relevant)
# NDCG rewards relevant docs appearing earlier
```

---

## Generation Metrics

### BLEU Score

```python
from nltk.translate.bleu_score import sentence_bleu

def calculate_bleu(reference, candidate):
    """Measures n-gram overlap with reference"""
    reference_tokens = [reference.split()]
    candidate_tokens = candidate.split()
    return sentence_bleu(reference_tokens, candidate_tokens)

# Example:
# Reference: "The cat sat on the mat"
# Candidate: "The cat is on the mat"
# BLEU ≈ 0.6 (some overlap but not exact)

# Limitations:
# - Doesn't understand meaning
# - Multiple valid answers score low
# - Not great for open-ended generation
```

### ROUGE Score

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

def calculate_rouge(reference, candidate):
    """Measures recall of n-grams"""
    scores = scorer.score(reference, candidate)
    return scores

# ROUGE-1: Unigram overlap
# ROUGE-2: Bigram overlap
# ROUGE-L: Longest common subsequence

# Best for: Summarization
```

### BERTScore

```python
from bert_score import score

def calculate_bertscore(references, candidates):
    """Semantic similarity using BERT embeddings"""
    P, R, F1 = score(candidates, references, lang="en")
    return F1.mean().item()

# Better than BLEU/ROUGE because:
# - Understands synonyms
# - Semantic similarity, not just word overlap
# - More aligned with human judgment
```

### Semantic Similarity

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_similarity(text1, text2):
    """Cosine similarity of embeddings"""
    emb1 = model.encode(text1)
    emb2 = model.encode(text2)
    return util.cos_sim(emb1, emb2).item()

# Example:
# "The dog is running" vs "A canine is jogging"
# Similarity ≈ 0.85 (same meaning, different words)
```

---

## LLM-as-Judge Evaluation

Use an LLM to assess another LLM's output.

### Basic LLM Judge

```python
def llm_judge(question, answer, reference=None):
    prompt = f"""Rate the following answer on a scale of 1-5:

Question: {question}
Answer: {answer}
{f'Reference Answer: {reference}' if reference else ''}

Criteria:
- Accuracy: Is the information correct?
- Relevance: Does it answer the question?
- Completeness: Is the answer thorough?
- Clarity: Is it easy to understand?

Provide:
1. Score (1-5)
2. Brief justification

Format: Score: X/5 | Reason: ..."""

    return llm.invoke(prompt)
```

### Pairwise Comparison

```python
def pairwise_comparison(question, answer_a, answer_b):
    prompt = f"""Compare these two answers and pick the better one:

Question: {question}

Answer A: {answer_a}

Answer B: {answer_b}

Which answer is better and why?
Reply with: "A is better because..." or "B is better because..." """

    return llm.invoke(prompt)
```

### G-Assessment Framework

```python
def g_assessment(question, answer, criteria):
    """
    Uses chain-of-thought for better assessment
    """
    prompt = f"""Assess the answer based on the criteria.
Think step by step before giving a score.

Question: {question}
Answer: {answer}

Criteria: {criteria}

Step-by-step assessment:
1. First, identify what the question is asking...
2. Then, check if the answer addresses this...
3. Check against the criteria...
4. Consider any issues or strengths...

Final Score (1-10): """

    return llm.invoke(prompt)
```

---

## Assessment Frameworks

### RAGAS

```python
from ragas import assess
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

# Prepare data
data = {
    "question": ["What is Python?"],
    "answer": ["Python is a programming language."],
    "contexts": [["Python is a high-level programming language..."]],
    "ground_truth": ["Python is a versatile programming language."]
}
dataset = Dataset.from_dict(data)

# Run assessment
results = assess(
    dataset,
    metrics=[
        faithfulness,        # Is answer grounded in context?
        answer_relevancy,    # Does answer address question?
        context_precision,   # Is context relevant?
        context_recall,      # Was all relevant context retrieved?
    ]
)

print(results)
# {'faithfulness': 0.95, 'answer_relevancy': 0.88, ...}
```

### DeepAssess

```python
from deepeval import assess
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase

# Create test case
test_case = LLMTestCase(
    input="What is machine learning?",
    actual_output="Machine learning is a subset of AI...",
    retrieval_context=["Machine learning involves..."],
    expected_output="ML is a type of artificial intelligence..."
)

# Define metrics
metrics = [
    AnswerRelevancyMetric(threshold=0.7),
    FaithfulnessMetric(threshold=0.7),
    HallucinationMetric(threshold=0.5),
]

# Run assessment
results = assess([test_case], metrics)
```

### TruLens

```python
from trulens_assess import TruChain, Feedback, Tru
from trulens_assess.feedback import Groundedness, AnswerRelevance

tru = Tru()

# Define feedback functions
groundedness = Groundedness()
answer_relevance = AnswerRelevance()

feedbacks = [
    Feedback(groundedness.groundedness_measure).on_output(),
    Feedback(answer_relevance.relevance).on_input_output(),
]

# Wrap your chain
tru_chain = TruChain(
    rag_chain,
    app_id="my_rag_app",
    feedbacks=feedbacks
)

# Run and assess
response = tru_chain("What is RAG?")

# View dashboard
tru.run_dashboard()
```

---

## Building an Assessment Pipeline

### Step 1: Create Test Dataset

```python
test_cases = [
    {
        "question": "What is the return policy?",
        "expected_answer": "30-day money-back guarantee",
        "relevant_docs": ["doc_123", "doc_456"],
        "category": "policy",
        "difficulty": "easy"
    },
    # ... more test cases
]

# Sources for test cases:
# 1. Historical user queries
# 2. Edge cases from production
# 3. Synthetic generation (LLM creates questions from docs)
# 4. Domain expert creation
```

### Step 2: Run Assessment

```python
def assess_rag_system(rag_chain, test_cases):
    results = []

    for case in test_cases:
        # Get RAG response
        response = rag_chain.invoke(case["question"])

        # Extract components
        answer = response["answer"]
        retrieved_docs = response["source_documents"]

        # Calculate metrics
        result = {
            "question": case["question"],
            "answer": answer,
            "expected": case["expected_answer"],

            # Retrieval metrics
            "retrieval_precision": calculate_precision(
                retrieved_docs, case["relevant_docs"]
            ),
            "retrieval_recall": calculate_recall(
                retrieved_docs, case["relevant_docs"]
            ),

            # Generation metrics
            "semantic_similarity": semantic_similarity(
                answer, case["expected_answer"]
            ),
            "faithfulness": check_faithfulness(answer, retrieved_docs),

            # LLM judge
            "llm_score": llm_judge(
                case["question"], answer, case["expected_answer"]
            ),
        }
        results.append(result)

    return results
```

### Step 3: Aggregate and Report

```python
def generate_report(results):
    report = {
        "total_cases": len(results),
        "avg_retrieval_precision": np.mean([r["retrieval_precision"] for r in results]),
        "avg_retrieval_recall": np.mean([r["retrieval_recall"] for r in results]),
        "avg_semantic_similarity": np.mean([r["semantic_similarity"] for r in results]),
        "avg_faithfulness": np.mean([r["faithfulness"] for r in results]),
        "avg_llm_score": np.mean([r["llm_score"] for r in results]),

        # By category
        "by_category": group_by_category(results),

        # Failure analysis
        "low_scoring_cases": [r for r in results if r["llm_score"] < 3],
    }
    return report
```

---

## Human Evaluation

### Annotation Guidelines

```
Rating Scale (1-5):

5 - Excellent
   - Completely answers the question
   - Accurate and well-sourced
   - Clear and concise

4 - Good
   - Mostly answers the question
   - Minor issues or omissions
   - Generally clear

3 - Acceptable
   - Partially answers the question
   - Some inaccuracies
   - Could be clearer

2 - Poor
   - Barely addresses the question
   - Significant errors
   - Confusing

1 - Unacceptable
   - Doesn't answer the question
   - Completely wrong
   - Incomprehensible
```

### Inter-Annotator Agreement

```python
from sklearn.metrics import cohen_kappa_score

def calculate_agreement(annotator1_scores, annotator2_scores):
    """Cohen's Kappa for inter-annotator agreement"""
    kappa = cohen_kappa_score(annotator1_scores, annotator2_scores)
    return kappa

# Interpretation:
# < 0.20: Poor
# 0.21-0.40: Fair
# 0.41-0.60: Moderate
# 0.61-0.80: Substantial
# 0.81-1.00: Almost perfect
```

---

## Online Evaluation

### A/B Testing

```python
def ab_test_router(user_id, variants):
    """Consistently assign users to variants"""
    hash_value = hash(user_id) % 100

    if hash_value < 50:
        return "control"  # Current system
    else:
        return "treatment"  # New system

# Metrics to track:
# - User satisfaction (thumbs up/down)
# - Task completion rate
# - Time to resolution
# - Follow-up questions (fewer = better)
```

### User Feedback Collection

```python
feedback_schema = {
    "response_id": "uuid",
    "rating": "1-5 or thumbs up/down",
    "feedback_text": "optional comment",
    "correction": "what should the answer be",
    "timestamp": "datetime",
    "user_id": "anonymized"
}

# Analyze feedback:
# 1. Track rating trends over time
# 2. Categorize negative feedback
# 3. Use corrections for fine-tuning
# 4. Identify systematic issues
```

---

## Assessment Checklist

### Before Deployment
- [ ] Test dataset with 100+ diverse examples
- [ ] Retrieval metrics (Precision, Recall, MRR)
- [ ] Generation metrics (Faithfulness, Relevance)
- [ ] Edge case testing
- [ ] Latency benchmarks
- [ ] Cost projections

### After Deployment
- [ ] User feedback mechanism
- [ ] Error logging and monitoring
- [ ] A/B testing framework
- [ ] Regular assessment runs
- [ ] Drift detection

---

## Interview Questions

**Q: How would you assess a RAG system?**
> Use the RAG Triad: Context Relevance (are right docs retrieved?), Groundedness (is answer based on context?), Answer Relevance (does it answer the question?). Combine automated metrics with LLM-as-judge and human assessment.

**Q: What's the problem with BLEU/ROUGE for LLM assessment?**
> They measure surface-level word overlap, not semantic meaning. "The dog ran quickly" vs "The canine sprinted fast" would score low despite same meaning. Use BERTScore or semantic similarity instead.

**Q: How do you handle assessment when there's no "correct" answer?**
> 1) Use pairwise comparison (which is better?)
> 2) LLM-as-judge with rubrics
> 3) Human assessment with clear guidelines
> 4) Multiple reference answers
> 5) Focus on criteria (helpful, harmless, honest)

**Q: How do you catch regressions in production?**
> 1) Continuous assessment on test set
> 2) Monitor user feedback trends
> 3) Sample and manually review responses
> 4) A/B test changes before full rollout
> 5) Automated alerts on metric drops
