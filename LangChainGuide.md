# LangChain Complete Guide

Everything you need to know about LangChain - the most popular framework for building LLM applications.

---

## What is LangChain?

LangChain is a framework for building applications powered by LLMs. It provides:
- **Chains**: Combine LLMs with other components
- **Agents**: LLMs that can use tools and make decisions
- **Memory**: Maintain conversation state
- **Retrieval**: Connect LLMs to your data (RAG)

---

## Installation

```bash
# Core package
pip install langchain

# Community integrations
pip install langchain-community

# Specific providers
pip install langchain-openai        # OpenAI
pip install langchain-anthropic     # Claude
pip install langchain-huggingface   # Hugging Face
pip install langchain-google-genai  # Google Gemini

# Vector stores
pip install chromadb faiss-cpu pinecone-client

# Document loaders
pip install pypdf unstructured
```

---

## Package Structure (v0.3+)

```
langchain              # Core abstractions
langchain-core         # Base interfaces
langchain-community    # Third-party integrations
langchain-openai       # OpenAI specific
langchain-anthropic    # Anthropic specific
langgraph              # Agent orchestration
langserve              # Deploy as API
langsmith              # Observability
```

---

## LLMs and Chat Models

### OpenAI

```python
from langchain_openai import ChatOpenAI, OpenAI

# Chat model (recommended)
chat = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,
    api_key="sk-xxxxx"  # or set OPENAI_API_KEY env var
)

# Invoke
response = chat.invoke("What is Python?")
print(response.content)

# With messages
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is Python?")
]
response = chat.invoke(messages)
```

### Anthropic (Claude)

```python
from langchain_anthropic import ChatAnthropic

chat = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    temperature=0.7,
    api_key="sk-ant-xxxxx"
)

response = chat.invoke("Explain quantum computing")
print(response.content)
```

### Ollama (Local)

```python
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama

# LLM
llm = Ollama(model="llama3")
response = llm.invoke("What is AI?")

# Chat model
chat = ChatOllama(model="llama3", temperature=0.7)
response = chat.invoke("What is AI?")
```

### Hugging Face

```python
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint

# Local model
from transformers import pipeline
pipe = pipeline("text-generation", model="gpt2", max_new_tokens=100)
llm = HuggingFacePipeline(pipeline=pipe)

# Inference API
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token="hf_xxxxx"
)
```

### Streaming

```python
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model="gpt-4", streaming=True)

# Stream responses
for chunk in chat.stream("Write a poem about coding"):
    print(chunk.content, end="", flush=True)
```

---

## Prompt Templates

### Basic Template

```python
from langchain_core.prompts import PromptTemplate

# Simple template
template = PromptTemplate.from_template(
    "What is a good name for a company that makes {product}?"
)
prompt = template.format(product="colorful socks")
# "What is a good name for a company that makes colorful socks?"

# With multiple variables
template = PromptTemplate(
    input_variables=["product", "style"],
    template="Suggest a {style} name for a {product} company."
)
prompt = template.format(product="shoes", style="funny")
```

### Chat Prompt Template

```python
from langchain_core.prompts import ChatPromptTemplate

# From messages
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that speaks like {persona}."),
    ("human", "{question}")
])

messages = template.format_messages(
    persona="a pirate",
    question="What is Python?"
)
```

### Few-Shot Prompting

```python
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "fast", "output": "slow"},
]

example_template = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}"
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_template,
    prefix="Give the opposite of each input:",
    suffix="Input: {word}\nOutput:",
    input_variables=["word"]
)

prompt = few_shot_prompt.format(word="big")
```

---

## Chains (LCEL - LangChain Expression Language)

### Basic Chain

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Components
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
model = ChatOpenAI(model="gpt-4")
parser = StrOutputParser()

# Chain with pipe operator
chain = prompt | model | parser

# Run
result = chain.invoke({"topic": "programming"})
print(result)
```

### Chain with Multiple Steps

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI()

# Step 1: Generate a topic
topic_prompt = ChatPromptTemplate.from_template(
    "Generate a random topic for a blog post about {subject}"
)

# Step 2: Write about the topic
blog_prompt = ChatPromptTemplate.from_template(
    "Write a short blog post about: {topic}"
)

# Chain them
topic_chain = topic_prompt | model | StrOutputParser()
blog_chain = blog_prompt | model | StrOutputParser()

# Full chain using RunnablePassthrough
from langchain_core.runnables import RunnablePassthrough

full_chain = (
    {"topic": topic_chain}
    | blog_chain
)

result = full_chain.invoke({"subject": "technology"})
```

### Parallel Chains

```python
from langchain_core.runnables import RunnableParallel

# Run multiple chains in parallel
parallel_chain = RunnableParallel(
    joke=prompt1 | model | parser,
    poem=prompt2 | model | parser,
    story=prompt3 | model | parser,
)

results = parallel_chain.invoke({"topic": "cats"})
# {'joke': '...', 'poem': '...', 'story': '...'}
```

### Branching (Router)

```python
from langchain_core.runnables import RunnableBranch

# Route based on input
branch = RunnableBranch(
    (lambda x: "math" in x["topic"].lower(), math_chain),
    (lambda x: "science" in x["topic"].lower(), science_chain),
    default_chain  # fallback
)

result = branch.invoke({"topic": "math problem"})
```

---

## Output Parsers

### String Parser

```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()
chain = prompt | model | parser
# Returns plain string
```

### JSON Parser

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# Define schema
class Person(BaseModel):
    name: str = Field(description="Person's name")
    age: int = Field(description="Person's age")

parser = JsonOutputParser(pydantic_object=Person)

prompt = PromptTemplate(
    template="Extract person info:\n{format_instructions}\n{text}",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | model | parser
result = chain.invoke({"text": "John is 25 years old"})
# {'name': 'John', 'age': 25}
```

### Structured Output (Pydantic)

```python
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

class MovieReview(BaseModel):
    title: str = Field(description="Movie title")
    rating: int = Field(description="Rating 1-10")
    summary: str = Field(description="Brief summary")

model = ChatOpenAI(model="gpt-4")
structured_model = model.with_structured_output(MovieReview)

result = structured_model.invoke("Review the movie Inception")
print(result.title)   # "Inception"
print(result.rating)  # 9
```

### List Parser

```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()

prompt = PromptTemplate(
    template="List 5 {category}.\n{format_instructions}",
    input_variables=["category"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | model | parser
result = chain.invoke({"category": "programming languages"})
# ['Python', 'JavaScript', 'Java', 'C++', 'Go']
```

---

## Document Loaders

### PDF

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("document.pdf")
pages = loader.load()

# Each page is a Document
for page in pages:
    print(page.page_content)
    print(page.metadata)  # {'source': 'document.pdf', 'page': 0}
```

### Directory of PDFs

```python
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    "./documents",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)
docs = loader.load()
```

### Text Files

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("file.txt")
docs = loader.load()
```

### CSV

```python
from langchain_community.document_loaders import CSVLoader

loader = CSVLoader("data.csv")
docs = loader.load()

# With specific columns
loader = CSVLoader(
    "data.csv",
    csv_args={"delimiter": ","},
    source_column="url"
)
```

### Web Pages

```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://example.com")
docs = loader.load()

# Multiple URLs
loader = WebBaseLoader(["https://url1.com", "https://url2.com"])
docs = loader.load()
```

### JSON

```python
from langchain_community.document_loaders import JSONLoader

loader = JSONLoader(
    file_path="data.json",
    jq_schema=".messages[].content",  # jq query
    text_content=False
)
docs = loader.load()
```

### Word Documents

```python
from langchain_community.document_loaders import Docx2txtLoader

loader = Docx2txtLoader("document.docx")
docs = loader.load()
```

### YouTube

```python
from langchain_community.document_loaders import YoutubeLoader

loader = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=xxxxx",
    add_video_info=True
)
docs = loader.load()
```

---

## Text Splitters

### Recursive Character Splitter (Recommended)

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

chunks = splitter.split_documents(docs)
# or
chunks = splitter.split_text("long text here...")
```

### Character Splitter

```python
from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)
```

### Token-based Splitter

```python
from langchain.text_splitter import TokenTextSplitter

splitter = TokenTextSplitter(
    chunk_size=100,      # tokens
    chunk_overlap=10
)

chunks = splitter.split_documents(docs)
```

### Markdown Splitter

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

headers = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
chunks = splitter.split_text(markdown_text)
```

### Code Splitter

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

# Python code
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=500,
    chunk_overlap=50
)

# Supports: PYTHON, JS, JAVA, GO, RUST, CPP, etc.
```

---

## Embeddings

### OpenAI

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key="sk-xxxxx"
)

# Single text
vector = embeddings.embed_query("Hello world")

# Multiple texts
vectors = embeddings.embed_documents(["Hello", "World"])
```

### Hugging Face

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

vector = embeddings.embed_query("Hello world")
```

### Cohere

```python
from langchain_cohere import CohereEmbeddings

embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key="xxxxx"
)
```

### Ollama

```python
from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="llama3")
vector = embeddings.embed_query("Hello world")
```

---

## Vector Stores

### ChromaDB

```python
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create from documents
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Load existing
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Add more documents
vectorstore.add_documents(new_docs)

# Search
results = vectorstore.similarity_search("query", k=3)
results_with_scores = vectorstore.similarity_search_with_score("query", k=3)
```

### FAISS

```python
from langchain_community.vectorstores import FAISS

# Create
vectorstore = FAISS.from_documents(chunks, embeddings)

# Save
vectorstore.save_local("faiss_index")

# Load
vectorstore = FAISS.load_local("faiss_index", embeddings)

# Search
results = vectorstore.similarity_search("query", k=3)

# MMR search (diversity)
results = vectorstore.max_marginal_relevance_search("query", k=3, fetch_k=10)
```

### Pinecone

```python
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

pc = Pinecone(api_key="xxxxx")
index = pc.Index("my-index")

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text"
)

# From documents
vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name="my-index"
)
```

### As Retriever

```python
# Convert to retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",      # or "mmr"
    search_kwargs={"k": 3}
)

# Use in chain
docs = retriever.invoke("query")
```

---

## Retrievers

### Basic Retriever

```python
# From vector store
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

docs = retriever.invoke("What is machine learning?")
```

### Multi-Query Retriever

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=ChatOpenAI()
)

# Generates multiple query variations
docs = retriever.invoke("What is RAG?")
```

### Contextual Compression

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(ChatOpenAI())

retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever()
)

# Returns only relevant parts of documents
docs = retriever.invoke("query")
```

### Parent Document Retriever

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

# Store for parent documents
store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# Add documents
retriever.add_documents(docs)

# Search retrieves parents
results = retriever.invoke("query")
```

### Ensemble Retriever

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# Keyword retriever
bm25 = BM25Retriever.from_documents(docs)

# Vector retriever
vector_retriever = vectorstore.as_retriever()

# Combine (hybrid search)
ensemble = EnsembleRetriever(
    retrievers=[bm25, vector_retriever],
    weights=[0.5, 0.5]
)

results = ensemble.invoke("query")
```

---

## RAG (Retrieval-Augmented Generation)

### Basic RAG Chain

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Components
retriever = vectorstore.as_retriever()
model = ChatOpenAI()

# Prompt
template = """Answer based on the context:

Context: {context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Format docs
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# Run
answer = rag_chain.invoke("What is machine learning?")
```

### RAG with Sources

```python
from langchain_core.runnables import RunnableParallel

# Chain that returns sources
rag_chain_with_sources = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(
    answer=lambda x: (
        prompt.format(context=format_docs(x["context"]), question=x["question"])
        | model
        | StrOutputParser()
    )
)

result = rag_chain_with_sources.invoke("What is RAG?")
print(result["answer"])
print(result["context"])  # Source documents
```

### RetrievalQA Chain (Legacy but Common)

```python
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    chain_type="stuff",  # stuff, map_reduce, refine, map_rerank
    retriever=retriever,
    return_source_documents=True
)

result = qa_chain.invoke({"query": "What is Python?"})
print(result["result"])
print(result["source_documents"])
```

---

## Memory

### Conversation Buffer Memory

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(return_messages=True)

# Add messages
memory.save_context(
    {"input": "Hi, I'm John"},
    {"output": "Hello John! How can I help you?"}
)

# Get history
history = memory.load_memory_variables({})
print(history["history"])
```

### Conversation Buffer Window

```python
from langchain.memory import ConversationBufferWindowMemory

# Only keep last k exchanges
memory = ConversationBufferWindowMemory(k=5, return_messages=True)
```

### Conversation Summary Memory

```python
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(
    llm=ChatOpenAI(),
    return_messages=True
)

# Summarizes old conversations to save tokens
```

### Conversation with Memory

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# Store for sessions
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Prompt with history
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | model

# Wrap with history
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Use with session
response = chain_with_history.invoke(
    {"input": "Hi, I'm Alice"},
    config={"configurable": {"session_id": "user123"}}
)

response = chain_with_history.invoke(
    {"input": "What's my name?"},
    config={"configurable": {"session_id": "user123"}}
)
# "Your name is Alice"
```

---

## Tools and Agents

### Built-in Tools

```python
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# Search
search = DuckDuckGoSearchRun()
result = search.invoke("What is LangChain?")

# Wikipedia
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
result = wiki.invoke("Python programming language")
```

### Custom Tools

```python
from langchain.tools import tool
from langchain_core.tools import Tool

# Using decorator
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

# Using Tool class
def search_db(query: str) -> str:
    """Search the database."""
    return f"Results for: {query}"

search_tool = Tool(
    name="database_search",
    description="Search the database for information",
    func=search_db
)
```

### Tool with Pydantic Schema

```python
from langchain.tools import StructuredTool
from langchain_core.pydantic_v1 import BaseModel, Field

class CalculatorInput(BaseModel):
    a: int = Field(description="First number")
    b: int = Field(description="Second number")
    operation: str = Field(description="Operation: add, subtract, multiply, divide")

def calculator(a: int, b: int, operation: str) -> float:
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        return a / b

calc_tool = StructuredTool.from_function(
    func=calculator,
    name="calculator",
    description="Perform math operations",
    args_schema=CalculatorInput
)
```

### Create Agent

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

# Tools
tools = [search, multiply, calc_tool]

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# Create agent
model = ChatOpenAI(model="gpt-4")
agent = create_tool_calling_agent(model, tools, prompt)

# Executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5
)

# Run
result = agent_executor.invoke({"input": "What is 25 * 4?"})
print(result["output"])
```

### ReAct Agent

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# Get ReAct prompt
prompt = hub.pull("hwchase17/react")

# Create agent
agent = create_react_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = agent_executor.invoke({"input": "Search for LangChain and summarize"})
```

---

## Callbacks and Logging

### Basic Callbacks

```python
from langchain_core.callbacks import StdOutCallbackHandler

handler = StdOutCallbackHandler()

# Use in chain
chain.invoke({"input": "Hello"}, config={"callbacks": [handler]})
```

### Custom Callback

```python
from langchain_core.callbacks import BaseCallbackHandler

class MyCallback(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"LLM started with: {prompts}")

    def on_llm_end(self, response, **kwargs):
        print(f"LLM response: {response}")

    def on_chain_start(self, serialized, inputs, **kwargs):
        print(f"Chain started: {inputs}")

    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"Tool called: {input_str}")

handler = MyCallback()
chain.invoke({"input": "Hello"}, config={"callbacks": [handler]})
```

### LangSmith Tracing

```python
import os

# Set environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls__xxxxx"
os.environ["LANGCHAIN_PROJECT"] = "my-project"

# All chains automatically traced
chain.invoke({"input": "Hello"})
```

---

## Caching

### In-Memory Cache

```python
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache

set_llm_cache(InMemoryCache())

# Same prompts return cached results
response1 = llm.invoke("Hello")  # Calls API
response2 = llm.invoke("Hello")  # Returns cached
```

### SQLite Cache

```python
from langchain.cache import SQLiteCache

set_llm_cache(SQLiteCache(database_path=".langchain.db"))
```

---

## Common Patterns

### Question Answering over Documents

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# 1. Load
loader = PyPDFLoader("document.pdf")
docs = loader.load()

# 2. Split
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. Embed & Store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings)

# 4. Create QA chain
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    retriever=vectorstore.as_retriever()
)

# 5. Ask
answer = qa.invoke("What is the main topic?")
```

### Summarization

```python
from langchain.chains.summarize import load_summarize_chain

# Stuff (small docs)
chain = load_summarize_chain(llm, chain_type="stuff")
summary = chain.invoke(docs)

# Map-Reduce (large docs)
chain = load_summarize_chain(llm, chain_type="map_reduce")
summary = chain.invoke(docs)

# Refine (iterative)
chain = load_summarize_chain(llm, chain_type="refine")
summary = chain.invoke(docs)
```

### Chatbot with RAG

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

template = ChatPromptTemplate.from_messages([
    ("system", "Answer based on context: {context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

chain = (
    {
        "context": retriever | format_docs,
        "question": lambda x: x["question"],
        "history": lambda x: x["history"]
    }
    | template
    | model
    | StrOutputParser()
)
```

---

## Quick Reference

### Common Imports

```python
# Models
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Ollama

# Prompts
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# Parsers
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

# Documents
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector Stores
from langchain_community.vectorstores import Chroma, FAISS

# Chains
from langchain.chains import RetrievalQA
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# Agents
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool

# Memory
from langchain.memory import ConversationBufferMemory
```

### Cheat Sheet

| Task | Code |
|------|------|
| Load PDF | `PyPDFLoader("file.pdf").load()` |
| Split text | `RecursiveCharacterTextSplitter(chunk_size=500).split_documents(docs)` |
| Create embeddings | `HuggingFaceEmbeddings().embed_query("text")` |
| Store vectors | `Chroma.from_documents(docs, embeddings)` |
| Search | `vectorstore.similarity_search("query", k=3)` |
| Create retriever | `vectorstore.as_retriever()` |
| Basic chain | `prompt \| model \| parser` |
| RAG chain | `{"context": retriever, "question": RunnablePassthrough()} \| prompt \| model` |
| Create tool | `@tool def my_tool(x): ...` |
| Create agent | `create_tool_calling_agent(model, tools, prompt)` |

---

## Environment Variables

```bash
# OpenAI
export OPENAI_API_KEY=sk-xxxxx

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-xxxxx

# Hugging Face
export HUGGINGFACEHUB_API_TOKEN=hf_xxxxx

# LangSmith (tracing)
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=ls__xxxxx
export LANGCHAIN_PROJECT=my-project
```
