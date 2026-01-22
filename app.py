"""
Simple RAG Chatbot using LangChain, Ollama, and ChromaDB
Perfect for interview demonstrations!
"""

import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ============== CONFIGURATION ==============
DOCUMENTS_PATH = "./documents"
CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "phi3:mini"  # You have this model installed. Change to "llama3" or "mistral" if you pull them

# ============== STEP 1: LOAD DOCUMENTS ==============
def load_documents():
    """Load all PDF documents from the documents folder"""
    print("📄 Loading PDF documents...")

    if not os.path.exists(DOCUMENTS_PATH):
        os.makedirs(DOCUMENTS_PATH)
        print(f"Created '{DOCUMENTS_PATH}' folder. Please add PDF files and restart.")
        return []

    loader = DirectoryLoader(
        DOCUMENTS_PATH,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages from PDFs")
    return documents

# ============== STEP 2: SPLIT INTO CHUNKS ==============
def split_documents(documents):
    """Split documents into smaller chunks for better retrieval"""
    print("✂️ Splitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # Size of each chunk
        chunk_overlap=50,    # Overlap between chunks
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks")
    return chunks

# ============== STEP 3: CREATE VECTOR STORE ==============
def create_vector_store(chunks):
    """Create embeddings and store in ChromaDB"""
    print("🔢 Creating embeddings and vector store...")

    # Initialize embedding model (runs locally, free!)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Create ChromaDB vector store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    print("✅ Vector store created and saved")
    return vector_store

# ============== STEP 4: LOAD EXISTING VECTOR STORE ==============
def load_vector_store():
    """Load existing vector store from disk"""
    print("📂 Loading existing vector store...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    print("✅ Vector store loaded")
    return vector_store

# ============== STEP 5: CREATE RAG CHAIN ==============
def create_rag_chain(vector_store):
    """Create the RAG chain with Ollama LLM"""
    print("🔗 Creating RAG chain...")

    # Initialize Ollama LLM (must be running locally)
    llm = Ollama(model=OLLAMA_MODEL, temperature=0.7)

    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}  # Return top 3 relevant chunks
    )

    # Custom prompt template
    prompt_template = """Use the following context to answer the question.
If you don't know the answer based on the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    print("✅ RAG chain ready")
    return qa_chain

# ============== MAIN APPLICATION ==============
def main():
    print("\n" + "="*50)
    print("🤖 RAG CHATBOT - Interview Demo")
    print("="*50 + "\n")

    # Check if vector store exists
    if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        print("Found existing vector store.")
        choice = input("Use existing (e) or rebuild (r)? [e/r]: ").strip().lower()

        if choice == 'r':
            # Rebuild from documents
            documents = load_documents()
            if not documents:
                return
            chunks = split_documents(documents)
            vector_store = create_vector_store(chunks)
        else:
            vector_store = load_vector_store()
    else:
        # First time setup
        documents = load_documents()
        if not documents:
            print("\n⚠️ No documents found!")
            print(f"Please add PDF files to '{DOCUMENTS_PATH}' folder and restart.")
            return
        chunks = split_documents(documents)
        vector_store = create_vector_store(chunks)

    # Create RAG chain
    qa_chain = create_rag_chain(vector_store)

    # Chat loop
    print("\n" + "="*50)
    print("💬 Chat with your documents!")
    print("Type 'quit' or 'exit' to stop")
    print("="*50 + "\n")

    while True:
        query = input("\n📝 Your question: ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break

        if not query:
            continue

        print("\n🔍 Searching and generating answer...")

        try:
            result = qa_chain.invoke({"query": query})

            print("\n" + "-"*40)
            print("📌 Answer:")
            print(result['result'])

            # Show sources (optional, great for interviews!)
            print("\n📚 Sources:")
            for i, doc in enumerate(result['source_documents'], 1):
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', 'N/A')
                print(f"  {i}. {source} (Page {page})")
            print("-"*40)

        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Make sure Ollama is running: 'ollama serve'")

if __name__ == "__main__":
    main()
