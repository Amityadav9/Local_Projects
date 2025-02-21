import os
import tempfile
import asyncio
from datetime import datetime
from typing import List, Optional
import streamlit as st
import bs4
from agno.agent import Agent
from agno.models.ollama import Ollama
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_core.embeddings import Embeddings
from agno.tools.exa import ExaTools
from agno.embedder.ollama import OllamaEmbedder
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.arxiv import ArxivTools


class OllamaEmbedderr(Embeddings):
    def __init__(self, model_name="snowflake-arctic-embed"):
        self.embedder = OllamaEmbedder(id=model_name, dimensions=1024)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.embedder.get_embedding(text)


# Constants
COLLECTION_NAME = "test-deepseek-r1"

# Streamlit App Initialization
st.title("🐋 Deepseek Local RAG Reasoning Agent")

# Session State Initialization
if "google_api_key" not in st.session_state:
    st.session_state.google_api_key = ""
if "qdrant_api_key" not in st.session_state:
    st.session_state.qdrant_api_key = ""
if "qdrant_url" not in st.session_state:
    st.session_state.qdrant_url = ""
if "use_local_qdrant" not in st.session_state:
    st.session_state.use_local_qdrant = True
if "model_version" not in st.session_state:
    st.session_state.model_version = "deepseek-r1:7b"
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_documents" not in st.session_state:
    st.session_state.processed_documents = []
if "history" not in st.session_state:
    st.session_state.history = []
if "use_web_search" not in st.session_state:
    st.session_state.use_web_search = True
if "force_web_search" not in st.session_state:
    st.session_state.force_web_search = False
if "similarity_threshold" not in st.session_state:
    st.session_state.similarity_threshold = 0.7
if "rag_enabled" not in st.session_state:
    st.session_state.rag_enabled = True
if "use_gpu" not in st.session_state:
    st.session_state.use_gpu = True

# Sidebar Configuration
st.sidebar.header("🤖 Agent Configuration")

# GPU Configuration
st.sidebar.header("🖥️ Hardware Configuration")
st.session_state.use_gpu = st.sidebar.toggle(
    "Use GPU Acceleration",
    value=st.session_state.use_gpu,
    help="Enable to use CUDA GPU acceleration for better performance GPU",
)

if st.session_state.use_gpu:
    gpu_info = "GPU detected. CUDA acceleration enabled."
    st.sidebar.success(gpu_info)
else:
    st.sidebar.warning("GPU acceleration disabled. Performance may be slower.")

# Model Selection
st.sidebar.header("📦 Model Selection")
model_help = """
Available Models:
- deepseek-r1:7b: High capability model
- qwen2.5-coder: Specialized for coding tasks
- llama3.2:3b: Lighter, faster model
- qwen2.5:7b: Well-rounded model
"""
st.session_state.model_version = st.sidebar.radio(
    "Select Model Version",
    options=["deepseek-r1:7b", "qwen2.5-coder:latest", "llama3.2:3b", "qwen2.5:7b"],
    help=model_help,
)
st.sidebar.info(
    f"Make sure you have pulled the selected model using: ollama pull {st.session_state.model_version}"
)

# RAG Mode Toggle
st.sidebar.header("🔍 RAG Configuration")
st.session_state.rag_enabled = st.sidebar.toggle(
    "Enable RAG Mode", value=st.session_state.rag_enabled
)

# Clear Chat Button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.history = []
    st.rerun()

# Show Qdrant Configuration only if RAG is enabled
if st.session_state.rag_enabled:
    st.sidebar.header("🔑 Vector Database Configuration")
    qdrant_mode = st.sidebar.radio(
        "Qdrant Connection Mode",
        options=["Local", "Cloud"],
        index=0 if st.session_state.use_local_qdrant else 1,
        help="Use local Docker instance or Qdrant Cloud",
    )

    st.session_state.use_local_qdrant = qdrant_mode == "Local"

    if not st.session_state.use_local_qdrant:
        qdrant_api_key = st.sidebar.text_input(
            "Qdrant API Key", type="password", value=st.session_state.qdrant_api_key
        )
        qdrant_url = st.sidebar.text_input(
            "Qdrant URL",
            placeholder="https://your-cluster.cloud.qdrant.io:6333",
            value=st.session_state.qdrant_url,
        )
        st.session_state.qdrant_api_key = qdrant_api_key
        st.session_state.qdrant_url = qdrant_url
    else:
        st.sidebar.success("Using local Qdrant at http://localhost:6333")
        st.session_state.qdrant_api_key = ""
        st.session_state.qdrant_url = ""

    st.sidebar.header("🎯 Search Configuration")
    st.session_state.similarity_threshold = st.sidebar.slider(
        "Document Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        help="Lower values will return more documents but might be less relevant.",
    )

# Web Search Configuration
st.sidebar.header("🌐 Web Search Configuration")
web_search_mode = st.sidebar.radio(
    "Web Search Mode",
    options=["Disabled", "Fallback Only", "Always Available"],
    help="""
    - Disabled: No web search
    - Fallback Only: Use web when no relevant documents found
    - Always Available: Can use web search anytime
    """,
)

st.session_state.use_web_search = web_search_mode != "Disabled"
st.session_state.web_search_fallback_only = web_search_mode == "Fallback Only"


def init_qdrant() -> Optional[QdrantClient]:
    try:
        if st.session_state.use_local_qdrant:
            return QdrantClient(url="http://localhost:6333", timeout=60)
        else:
            if not all([st.session_state.qdrant_api_key, st.session_state.qdrant_url]):
                st.warning("⚠️ Qdrant cloud configuration incomplete")
                return None
            return QdrantClient(
                url=st.session_state.qdrant_url,
                api_key=st.session_state.qdrant_api_key,
                timeout=60,
            )
    except Exception as e:
        st.error(f"🔴 Qdrant connection failed: {str(e)}")
        if st.session_state.use_local_qdrant:
            st.info("Make sure your Docker container is running: docker ps")
        return None


async def process_pdf_async(file) -> List:
    loop = asyncio.get_event_loop()

    def _process_pdf():
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file.getvalue())
                loader = PyPDFLoader(tmp_file.name)
                documents = loader.load()

                for doc in documents:
                    doc.metadata.update(
                        {
                            "source_type": "pdf",
                            "file_name": file.name,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200
                )
                return text_splitter.split_documents(documents)
        except Exception as e:
            st.error(f"📄 PDF processing error: {str(e)}")
            return []

    return await loop.run_in_executor(None, _process_pdf)


async def process_web_async(url: str) -> List:
    loop = asyncio.get_event_loop()

    def _process_web():
        try:
            loader = WebBaseLoader(
                web_paths=(url,),
                bs_kwargs=dict(
                    parse_only=bs4.SoupStrainer(
                        class_=(
                            "post-content",
                            "post-title",
                            "post-header",
                            "content",
                            "main",
                        )
                    )
                ),
            )
            documents = loader.load()

            for doc in documents:
                doc.metadata.update(
                    {
                        "source_type": "url",
                        "url": url,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            return text_splitter.split_documents(documents)
        except Exception as e:
            st.error(f"🌐 Web processing error: {str(e)}")
            return []

    return await loop.run_in_executor(None, _process_web)


async def create_vector_store_async(client, texts):
    loop = asyncio.get_event_loop()

    def _create_store():
        try:
            try:
                client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
                )
                return f"📚 Created new collection: {COLLECTION_NAME}"
            except Exception as e:
                if "already exists" not in str(e).lower():
                    raise e
                return None
        except Exception as e:
            raise e

    def _add_documents(vector_store):
        try:
            vector_store.add_documents(texts)
            return vector_store
        except Exception as e:
            raise e

    try:
        creation_msg = await loop.run_in_executor(None, _create_store)
        if creation_msg:
            st.success(creation_msg)

        vector_store = QdrantVectorStore(
            client=client, collection_name=COLLECTION_NAME, embedding=OllamaEmbedderr()
        )

        with st.spinner("📤 Uploading documents to Qdrant..."):
            vector_store = await loop.run_in_executor(
                None, lambda: _add_documents(vector_store)
            )
            st.success("✅ Documents stored successfully!")
            return vector_store

    except Exception as e:
        st.error(f"🔴 Vector store error: {str(e)}")
        return None


def get_web_search_agent() -> Optional[Agent]:
    """Initialize a web search agent using DuckDuckGo."""
    try:
        return Agent(
            name="Web Search Agent",
            model=Ollama(id="llama3.2:3b"),  # Use the lightweight model for web search
            tools=[
                DuckDuckGoTools(news_max_results=10)
            ],  # Simple implementation without extra arguments
            instructions="""You are a web search expert. Search the web and provide accurate, up-to-date information about the query.
            Also please Include sources in your response.""",
            show_tool_calls=True,
            markdown=True,
        )
    except Exception as e:
        st.error(f"Failed to initialize web search agent: {str(e)}")
        return None


async def perform_web_search(query: str) -> Optional[str]:
    """Perform web search with error handling."""
    try:
        agent = get_web_search_agent()
        if not agent:
            return None

        result = await agent.arun(query)
        return result.content if result else None
    except Exception as e:
        st.error(f"Web search error: {str(e)}")
        return None


def get_rag_agent() -> Agent:
    return Agent(
        name="DeepSeek RAG Agent",
        model=Ollama(id=st.session_state.model_version),
        instructions=[
            "You are an Intelligent Agent specializing in providing accurate answers.",
            "When answering questions:",
            "1. Analyze the question carefully",
            "2. Use provided context from documents when available",
            "3. For web search results, clearly indicate the source",
            "4. Maintain high accuracy and clarity",
            "5. Present information in a structured manner",
        ],
        show_tool_calls=True,
        markdown=True,
    )


async def check_document_relevance_async(
    query: str, vector_store, threshold: float = 0.7
) -> tuple[bool, List]:
    if not vector_store:
        return False, []

    loop = asyncio.get_event_loop()

    def _query_retriever():
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 5, "score_threshold": threshold},
        )
        return retriever.invoke(query)

    # Run vector search in a thread pool
    docs = await loop.run_in_executor(None, _query_retriever)
    return bool(docs), docs


async def generate_rag_response(query, rewritten_query, vector_store):
    """Generate RAG response with web search when needed."""
    context = ""
    docs = []

    # Determine search strategy
    force_web = st.session_state.force_web_search
    use_web = force_web or st.session_state.use_web_search

    # Try web search if enabled
    if use_web:
        with st.spinner("🔍 Searching the web..."):
            web_results = await perform_web_search(rewritten_query)
            if web_results:
                context = f"Web Search Results:\n{web_results}"
                st.info("ℹ️ Using web search results")

    # Try document search if no web results
    if not context and vector_store:
        has_docs, docs = await check_document_relevance_async(
            rewritten_query, vector_store, st.session_state.similarity_threshold
        )
        if has_docs:
            context = "\n\n".join([d.page_content for d in docs])
            st.info(f"📊 Found {len(docs)} relevant documents")

    # Generate response
    with st.spinner("🤖 Thinking..."):
        try:
            rag_agent = get_rag_agent()

            if context:
                full_prompt = f"""Context: {context}\n\nQuestion: {query}\nPlease provide a comprehensive answer based on the available information."""
            else:
                full_prompt = query
                st.info("ℹ️ Answering based on model knowledge only.")

            response = await rag_agent.arun(full_prompt)
            return response, docs

        except Exception as e:
            st.error(f"❌ Error generating response: {str(e)}")
            return None, []


async def generate_simple_response(query):
    """Generate simple response asynchronously (non-RAG mode)."""
    try:
        rag_agent = get_rag_agent()
        web_search_agent = (
            get_web_search_agent() if st.session_state.use_web_search else None
        )

        # Handle web search if forced or enabled
        context = ""
        if st.session_state.force_web_search and web_search_agent:
            with st.spinner("🔍 Searching the web..."):
                try:
                    web_results = await web_search_agent.arun(query)
                    if web_results and web_results.content:
                        context = f"Web Search Results:\n{web_results.content}"
                        st.info("ℹ️ Using web search as requested.")
                except Exception as e:
                    st.error(f"❌ Web search error: {str(e)}")

        # Generate response
        if context:
            full_prompt = f"""Context: {context}

Question: {query}

Please provide a comprehensive answer based on the available information."""
        else:
            full_prompt = query

        response = await rag_agent.arun(full_prompt)
        return response

    except Exception as e:
        st.error(f"❌ Error generating response: {str(e)}")
        return None


chat_col, toggle_col = st.columns([0.9, 0.1])

with chat_col:
    prompt = st.chat_input(
        "Ask about your documents..."
        if st.session_state.rag_enabled
        else "Ask me anything..."
    )

with toggle_col:
    st.session_state.force_web_search = st.toggle("🌐", help="Force web search")

# Check if RAG is enabled
if st.session_state.rag_enabled:
    qdrant_client = init_qdrant()

    # File/URL Upload Section
    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
    web_url = st.sidebar.text_input("Or enter URL")

    # Process documents (synchronous part - file uploads need to remain synchronous due to Streamlit limitations)
    if uploaded_file:
        file_name = uploaded_file.name
        if file_name not in st.session_state.processed_documents:
            with st.spinner("Processing PDF..."):
                # Use asyncio.run since we're in a synchronous context
                texts = asyncio.run(process_pdf_async(uploaded_file))
                if texts and qdrant_client:
                    if st.session_state.vector_store:
                        st.session_state.vector_store.add_documents(texts)
                    else:
                        st.session_state.vector_store = asyncio.run(
                            create_vector_store_async(qdrant_client, texts)
                        )
                    st.session_state.processed_documents.append(file_name)
                    st.success(f"✅ Added PDF: {file_name}")

    if web_url:
        if web_url not in st.session_state.processed_documents:
            with st.spinner("Processing URL..."):
                # Use asyncio.run since we're in a synchronous context
                texts = asyncio.run(process_web_async(web_url))
                if texts and qdrant_client:
                    if st.session_state.vector_store:
                        st.session_state.vector_store.add_documents(texts)
                    else:
                        st.session_state.vector_store = asyncio.run(
                            create_vector_store_async(qdrant_client, texts)
                        )
                    st.session_state.processed_documents.append(web_url)
                    st.success(f"✅ Added URL: {web_url}")

    # Display sources in sidebar
    if st.session_state.processed_documents:
        st.sidebar.header("📚 Processed Sources")
        for source in st.session_state.processed_documents:
            if source.endswith(".pdf"):
                st.sidebar.text(f"📄 {source}")
            else:
                st.sidebar.text(f"🌐 {source}")

# Main chat logic
if prompt:
    # Add user message to history
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    if st.session_state.rag_enabled:
        # RAG workflow
        with st.spinner("🤔 Evaluating the Query..."):
            try:
                rewritten_query = prompt

                with st.expander("Evaluating the query"):
                    st.write(f"User's Prompt: {prompt}")
            except Exception as e:
                st.error(f"❌ Error rewriting query: {str(e)}")
                rewritten_query = prompt

        # Process RAG response asynchronously
        response, docs = asyncio.run(
            generate_rag_response(
                prompt, rewritten_query, st.session_state.vector_store
            )
        )

        if response:
            # Add assistant response to history
            st.session_state.history.append(
                {"role": "assistant", "content": response.content}
            )

            # Display assistant response
            with st.chat_message("assistant"):
                st.write(response.content)

                # Show sources if available
                if not st.session_state.force_web_search and docs:
                    with st.expander("🔍 See document sources"):
                        for i, doc in enumerate(docs, 1):
                            source_type = doc.metadata.get("source_type", "unknown")
                            source_icon = "📄" if source_type == "pdf" else "🌐"
                            source_name = doc.metadata.get(
                                "file_name" if source_type == "pdf" else "url",
                                "unknown",
                            )
                            st.write(f"{source_icon} Source {i} from {source_name}:")
                            st.write(f"{doc.page_content[:200]}...")

    else:
        # Simple mode without RAG
        response = asyncio.run(generate_simple_response(prompt))

        if response:
            response_content = response.content

            # Extract thinking process and final response
            import re

            think_pattern = r"<think>(.*?)</think>"
            think_match = re.search(think_pattern, response_content, re.DOTALL)

            if think_match:
                thinking_process = think_match.group(1).strip()
                final_response = re.sub(
                    think_pattern, "", response_content, flags=re.DOTALL
                ).strip()
            else:
                thinking_process = None
                final_response = response_content

            # Add assistant response to history (only the final response)
            st.session_state.history.append(
                {"role": "assistant", "content": final_response}
            )

            # Display assistant response
            with st.chat_message("assistant"):
                if thinking_process:
                    with st.expander("🤔 See thinking process"):
                        st.markdown(thinking_process)
                st.markdown(final_response)

else:
    st.warning(
        "You can directly talk to r1 locally! Toggle the RAG mode to upload documents!"
    )


# Environment setup instructions for GPU acceleration
if st.session_state.use_gpu:
    with st.sidebar.expander("📝 GPU Setup Instructions"):
        st.markdown("""
        ### Optimizing Ollama for CUDA GPU 
        
        1. Make sure you have the NVIDIA drivers installed
        2. Start Ollama with GPU access:
        ```bash
        OLLAMA_USE_CUDA=1 ollama serve
        ```
        
        3. For Docker, ensure GPU passthrough:
        ```bash
        docker run -d --gpus all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
        ```
        
        4. Verify GPU usage:
        ```bash
        nvidia-smi
        ```
        
        GPU should significantly improve inference speed!
        """)
