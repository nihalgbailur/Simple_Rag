# Hugging Face Complete Guide

Everything you need to know about Hugging Face - from basic usage to advanced techniques.

---

## What is Hugging Face?

Hugging Face is the **GitHub for AI/ML** - a platform providing:
- **Model Hub**: 500K+ pre-trained models
- **Datasets Hub**: 100K+ datasets
- **Spaces**: Host ML demos
- **Libraries**: transformers, datasets, accelerate, etc.

---

## Ways to Use Hugging Face

### 1. Direct from Hub (Easiest)

```python
from transformers import pipeline

# One line to load and use any model
classifier = pipeline("sentiment-analysis")
result = classifier("I love this product!")
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

### 2. Via Inference API (No GPU needed)

```python
import requests

API_URL = "https://api-inference.huggingface.co/models/gpt2"
headers = {"Authorization": "Bearer YOUR_HF_TOKEN"}

response = requests.post(API_URL, headers=headers, json={"inputs": "Hello, I am"})
print(response.json())
```

### 3. Local Download & Use

```python
from transformers import AutoModel, AutoTokenizer

# Downloads model to ~/.cache/huggingface
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
```

### 4. Hugging Face Hub Library

```python
from huggingface_hub import hf_hub_download, snapshot_download

# Download single file
file_path = hf_hub_download(repo_id="gpt2", filename="config.json")

# Download entire model
model_path = snapshot_download(repo_id="gpt2")
```

### 5. Sentence Transformers (for Embeddings)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(["Hello world", "How are you"])
```

---

## Core Libraries

| Library | Purpose | Install |
|---------|---------|---------|
| `transformers` | Load & use models | `pip install transformers` |
| `datasets` | Load & process datasets | `pip install datasets` |
| `accelerate` | Distributed training | `pip install accelerate` |
| `huggingface_hub` | Interact with Hub | `pip install huggingface_hub` |
| `tokenizers` | Fast tokenization | `pip install tokenizers` |
| `sentence-transformers` | Embedding models | `pip install sentence-transformers` |
| `peft` | Efficient fine-tuning | `pip install peft` |
| `trl` | RLHF training | `pip install trl` |
| `evaluate` | Model evaluation | `pip install evaluate` |

---

## Pipeline - The Easiest Way

Pipeline is a high-level API that handles everything automatically.

### Available Pipelines

```python
from transformers import pipeline

# Text Classification
classifier = pipeline("text-classification")
classifier("I love Hugging Face!")

# Named Entity Recognition
ner = pipeline("ner", grouped_entities=True)
ner("My name is John and I work at Google in New York")

# Question Answering
qa = pipeline("question-answering")
qa(question="What is my name?", context="My name is John and I am a developer")

# Summarization
summarizer = pipeline("summarization")
summarizer("Long article text here...", max_length=100, min_length=30)

# Translation
translator = pipeline("translation_en_to_fr")
translator("Hello, how are you?")

# Text Generation
generator = pipeline("text-generation", model="gpt2")
generator("Once upon a time", max_length=50)

# Fill Mask
fill_mask = pipeline("fill-mask")
fill_mask("Paris is the [MASK] of France")

# Zero-Shot Classification
zero_shot = pipeline("zero-shot-classification")
zero_shot("I love playing football", candidate_labels=["sports", "politics", "food"])

# Sentiment Analysis
sentiment = pipeline("sentiment-analysis")
sentiment("This movie was amazing!")

# Feature Extraction (Embeddings)
extractor = pipeline("feature-extraction")
embeddings = extractor("Hello world")

# Image Classification
img_classifier = pipeline("image-classification")
img_classifier("path/to/image.jpg")

# Object Detection
detector = pipeline("object-detection")
detector("path/to/image.jpg")

# Speech Recognition
asr = pipeline("automatic-speech-recognition")
asr("path/to/audio.wav")

# Text-to-Speech
tts = pipeline("text-to-speech", model="microsoft/speecht5_tts")
audio = tts("Hello, this is a test")
```

### Pipeline with Specific Model

```python
# Use a specific model
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Use specific device
classifier = pipeline("text-classification", device=0)  # GPU 0
classifier = pipeline("text-classification", device=-1)  # CPU
```

---

## Auto Classes - Flexible Loading

Auto classes automatically detect and load the right model architecture.

### AutoTokenizer

```python
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize text
tokens = tokenizer("Hello, how are you?")
print(tokens)
# {'input_ids': [101, 7592, 1010, 2129, 2024, 2017, 1029, 102],
#  'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]}

# Decode back to text
text = tokenizer.decode(tokens['input_ids'])

# Batch tokenization
batch = tokenizer(["Hello", "World"], padding=True, truncation=True, return_tensors="pt")

# Get vocabulary size
vocab_size = tokenizer.vocab_size
```

### AutoModel

```python
from transformers import AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM

# Base model (no head)
model = AutoModel.from_pretrained("bert-base-uncased")

# Classification model
classifier = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Text generation model
generator = AutoModelForCausalLM.from_pretrained("gpt2")

# Question answering
qa_model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
```

### AutoConfig

```python
from transformers import AutoConfig

# Load config
config = AutoConfig.from_pretrained("bert-base-uncased")

print(config.hidden_size)       # 768
print(config.num_hidden_layers) # 12
print(config.num_attention_heads) # 12

# Modify config
config.num_labels = 5
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", config=config)
```

---

## Tokenizer Deep Dive

### Basic Operations

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Encode text to IDs
input_ids = tokenizer.encode("Hello world")

# Decode IDs to text
text = tokenizer.decode(input_ids)

# Tokenize (returns dict)
encoded = tokenizer("Hello world", return_tensors="pt")

# Access tokens as strings
tokens = tokenizer.tokenize("Hello world")
# ['hello', 'world']
```

### Advanced Tokenization

```python
# Padding and truncation
encoded = tokenizer(
    "Hello world",
    padding="max_length",      # Pad to max_length
    max_length=512,            # Maximum length
    truncation=True,           # Truncate if too long
    return_tensors="pt"        # Return PyTorch tensors
)

# Batch tokenization with different lengths
texts = ["Short text", "This is a much longer piece of text that needs padding"]
encoded = tokenizer(
    texts,
    padding=True,              # Pad to longest in batch
    truncation=True,
    return_tensors="pt"
)

# Get attention mask (1 for real tokens, 0 for padding)
attention_mask = encoded['attention_mask']

# Special tokens
print(tokenizer.cls_token)     # [CLS]
print(tokenizer.sep_token)     # [SEP]
print(tokenizer.pad_token)     # [PAD]
print(tokenizer.unk_token)     # [UNK]
print(tokenizer.mask_token)    # [MASK]
```

### Tokenizer for Chat Models

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
]

# Apply chat template
prompt = tokenizer.apply_chat_template(messages, tokenize=False)
```

---

## Model Inference

### Basic Inference

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Prepare input
inputs = tokenizer("I love this!", return_tensors="pt")

# Run inference
with torch.no_grad():
    outputs = model(**inputs)

# Get predictions
predictions = torch.softmax(outputs.logits, dim=-1)
print(predictions)  # tensor([[0.0002, 0.9998]])
```

### Text Generation

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

input_ids = tokenizer.encode("Hello, I am", return_tensors="pt")

# Generate
output = model.generate(
    input_ids,
    max_length=50,
    num_return_sequences=1,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
```

### Generation Parameters

```python
output = model.generate(
    input_ids,
    # Length control
    max_length=100,              # Maximum total length
    max_new_tokens=50,           # Maximum new tokens to generate
    min_length=10,               # Minimum length

    # Sampling strategy
    do_sample=True,              # Enable sampling (vs greedy)
    temperature=0.7,             # Randomness (lower = more focused)
    top_k=50,                    # Sample from top k tokens
    top_p=0.95,                  # Nucleus sampling

    # Beam search
    num_beams=5,                 # Beam search width
    early_stopping=True,         # Stop when all beams finish

    # Repetition control
    repetition_penalty=1.2,      # Penalize repeated tokens
    no_repeat_ngram_size=2,      # Prevent repeating n-grams

    # Output control
    num_return_sequences=3,      # Generate multiple sequences
    return_dict_in_generate=True,
    output_scores=True,
)
```

---

## Datasets Library

### Loading Datasets

```python
from datasets import load_dataset

# Load from Hub
dataset = load_dataset("imdb")
dataset = load_dataset("squad")
dataset = load_dataset("glue", "mrpc")  # With config

# Load specific split
train_data = load_dataset("imdb", split="train")
test_data = load_dataset("imdb", split="test")

# Load subset
small_data = load_dataset("imdb", split="train[:1000]")

# Load from local files
dataset = load_dataset("csv", data_files="my_data.csv")
dataset = load_dataset("json", data_files="my_data.json")
dataset = load_dataset("text", data_files="my_data.txt")

# Load from folder
dataset = load_dataset("imagefolder", data_dir="path/to/images")
```

### Dataset Operations

```python
from datasets import load_dataset

dataset = load_dataset("imdb", split="train")

# View structure
print(dataset)
print(dataset.features)
print(dataset.column_names)

# Access data
print(dataset[0])                    # First example
print(dataset["text"][:5])           # First 5 texts
print(dataset[10:20])                # Slice

# Filter
positive = dataset.filter(lambda x: x["label"] == 1)

# Map (apply function)
def tokenize(example):
    return tokenizer(example["text"], truncation=True)

tokenized = dataset.map(tokenize, batched=True)

# Select columns
dataset = dataset.select_columns(["text", "label"])

# Rename columns
dataset = dataset.rename_column("text", "input_text")

# Shuffle
dataset = dataset.shuffle(seed=42)

# Train/test split
dataset = dataset.train_test_split(test_size=0.2)

# Sort
dataset = dataset.sort("label")
```

### Streaming Large Datasets

```python
from datasets import load_dataset

# Stream instead of downloading
dataset = load_dataset("c4", "en", split="train", streaming=True)

# Iterate
for example in dataset:
    print(example)
    break

# Take first n examples
small = dataset.take(1000)
```

---

## Sentence Transformers (Embeddings)

### Basic Usage

```python
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Single sentence
embedding = model.encode("Hello world")
print(embedding.shape)  # (384,)

# Multiple sentences
sentences = ["Hello world", "How are you", "Nice to meet you"]
embeddings = model.encode(sentences)
print(embeddings.shape)  # (3, 384)
```

### Similarity Search

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

# Corpus
corpus = [
    "A man is eating food.",
    "A man is eating pasta.",
    "The girl is carrying a baby.",
    "A woman is playing violin.",
]
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

# Query
query = "A man is eating dinner"
query_embedding = model.encode(query, convert_to_tensor=True)

# Find similar
cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
top_results = torch.topk(cos_scores, k=2)

for score, idx in zip(top_results.values, top_results.indices):
    print(f"{corpus[idx]} (Score: {score:.4f})")
```

### Popular Embedding Models

| Model | Dimensions | Speed | Quality | Use Case |
|-------|------------|-------|---------|----------|
| `all-MiniLM-L6-v2` | 384 | Fast | Good | General purpose |
| `all-mpnet-base-v2` | 768 | Medium | Better | Higher quality needs |
| `multi-qa-MiniLM-L6-cos-v1` | 384 | Fast | Good | Q&A, retrieval |
| `paraphrase-MiniLM-L6-v2` | 384 | Fast | Good | Paraphrase detection |
| `all-MiniLM-L12-v2` | 384 | Medium | Better | Balance speed/quality |

### Semantic Search with FAISS

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings
corpus = ["Document 1 text", "Document 2 text", "Document 3 text"]
embeddings = model.encode(corpus)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype('float32'))

# Search
query = "search query"
query_vector = model.encode([query]).astype('float32')
distances, indices = index.search(query_vector, k=2)
```

---

## Hugging Face Hub

### Authentication

```python
from huggingface_hub import login

# Login (interactive)
login()

# Login with token
login(token="hf_xxxxx")

# Or set environment variable
# export HF_TOKEN=hf_xxxxx
```

### Download Models

```python
from huggingface_hub import hf_hub_download, snapshot_download

# Download single file
config_path = hf_hub_download(
    repo_id="bert-base-uncased",
    filename="config.json"
)

# Download entire model
model_path = snapshot_download(
    repo_id="bert-base-uncased",
    cache_dir="./models"
)

# Download specific revision
model_path = snapshot_download(
    repo_id="bert-base-uncased",
    revision="main"  # or commit hash
)
```

### Upload to Hub

```python
from huggingface_hub import HfApi, create_repo

api = HfApi()

# Create repository
create_repo("my-awesome-model", private=True)

# Upload file
api.upload_file(
    path_or_fileobj="model.bin",
    path_in_repo="model.bin",
    repo_id="username/my-awesome-model"
)

# Upload folder
api.upload_folder(
    folder_path="./my_model",
    repo_id="username/my-awesome-model"
)

# Using transformers
model.push_to_hub("my-awesome-model")
tokenizer.push_to_hub("my-awesome-model")
```

### Model Cards

```python
from huggingface_hub import ModelCard

# Load model card
card = ModelCard.load("bert-base-uncased")
print(card.content)

# Create model card
card = ModelCard.from_template(
    card_data=ModelCardData(
        language="en",
        license="mit",
        library_name="transformers",
        tags=["text-classification"],
    ),
    model_id="my-model",
    model_description="A fine-tuned BERT model",
)
card.push_to_hub("username/my-model")
```

---

## Loading Models Efficiently

### Half Precision (FP16)

```python
import torch
from transformers import AutoModelForCausalLM

# Load in FP16
model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    torch_dtype=torch.float16,
    device_map="auto"
)
```

### 8-bit Quantization

```python
from transformers import AutoModelForCausalLM

# Requires: pip install bitsandbytes accelerate
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,
    device_map="auto"
)
```

### 4-bit Quantization

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)
```

### Device Mapping

```python
from transformers import AutoModelForCausalLM

# Auto device map (spreads across GPUs)
model = AutoModelForCausalLM.from_pretrained(
    "large-model",
    device_map="auto"
)

# Specific device map
model = AutoModelForCausalLM.from_pretrained(
    "large-model",
    device_map={"": 0}  # All on GPU 0
)

# CPU offload for large models
model = AutoModelForCausalLM.from_pretrained(
    "very-large-model",
    device_map="auto",
    offload_folder="offload"
)
```

---

## Fine-Tuning with PEFT (LoRA)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_4bit=True,
    device_map="auto"
)

# Prepare for training
model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(
    r=16,                      # Rank
    lora_alpha=32,             # Scaling
    target_modules=["q_proj", "v_proj"],  # Which layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Check trainable parameters
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.062

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
)

# Train
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=512,
)

trainer.train()

# Save adapter
model.save_pretrained("./lora-adapter")

# Load adapter later
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = PeftModel.from_pretrained(base_model, "./lora-adapter")
```

---

## Evaluation

```python
from evaluate import load

# Load metrics
accuracy = load("accuracy")
f1 = load("f1")
bleu = load("bleu")
rouge = load("rouge")

# Compute accuracy
results = accuracy.compute(predictions=[0, 1, 1], references=[0, 1, 0])
print(results)  # {'accuracy': 0.666}

# F1 score
results = f1.compute(predictions=[0, 1, 1, 0], references=[0, 1, 0, 0])
print(results)  # {'f1': 0.666}

# BLEU (for translation)
predictions = ["hello there general kenobi"]
references = [["hello there general kenobi"]]
results = bleu.compute(predictions=predictions, references=references)

# ROUGE (for summarization)
predictions = ["hello there"]
references = ["hello there"]
results = rouge.compute(predictions=predictions, references=references)
```

---

## Integration with LangChain

### Embeddings

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Use with vector store
text_embedding = embeddings.embed_query("Hello world")
doc_embeddings = embeddings.embed_documents(["Doc 1", "Doc 2"])
```

### LLMs

```python
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

# Local pipeline
pipe = pipeline("text-generation", model="gpt2", max_new_tokens=100)
llm = HuggingFacePipeline(pipeline=pipe)

response = llm.invoke("Hello, how are you?")
```

### Inference API

```python
from langchain_huggingface import HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token="hf_xxxxx",
    temperature=0.7,
    max_new_tokens=512
)

response = llm.invoke("What is machine learning?")
```

---

## Quick Reference: Common Tasks

| Task | Code |
|------|------|
| **Load tokenizer** | `AutoTokenizer.from_pretrained("model")` |
| **Load model** | `AutoModel.from_pretrained("model")` |
| **Quick inference** | `pipeline("task")("input")` |
| **Get embeddings** | `SentenceTransformer("model").encode("text")` |
| **Load dataset** | `load_dataset("name")` |
| **Download model** | `snapshot_download("repo_id")` |
| **Push to Hub** | `model.push_to_hub("name")` |
| **4-bit loading** | `from_pretrained(..., load_in_4bit=True)` |

---

## Popular Models by Task

### Text Generation
- `gpt2`, `meta-llama/Llama-2-7b-hf`, `mistralai/Mistral-7B-v0.1`

### Embeddings
- `sentence-transformers/all-MiniLM-L6-v2`, `BAAI/bge-small-en-v1.5`

### Classification
- `distilbert-base-uncased-finetuned-sst-2-english`

### Question Answering
- `deepset/roberta-base-squad2`

### Summarization
- `facebook/bart-large-cnn`, `t5-base`

### Translation
- `Helsinki-NLP/opus-mt-en-de`

### Image
- `openai/clip-vit-base-patch32`, `stabilityai/stable-diffusion-2-1`

---

## Environment Variables

```bash
# Hugging Face token
export HF_TOKEN=hf_xxxxx

# Cache directory
export HF_HOME=/path/to/cache
export TRANSFORMERS_CACHE=/path/to/cache

# Offline mode
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Disable telemetry
export HF_HUB_DISABLE_TELEMETRY=1
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Use `load_in_8bit=True` or `load_in_4bit=True` |
| Slow download | Set `HF_HUB_ENABLE_HF_TRANSFER=1` |
| Model not found | Check model name on huggingface.co |
| Token required | Login with `huggingface-cli login` |
| CUDA error | Use `device_map="auto"` or `device="cpu"` |
| Tokenizer warning | Set `tokenizer.pad_token = tokenizer.eos_token` |
