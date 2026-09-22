import os
import shutil
import tempfile
import streamlit as st
import pypdf
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma

# Configuration
load_dotenv(override=True)
api_key = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
CHROMA_DIR = "./chroma_db"

# Page Configuration
st.set_page_config(page_title="PDFMind", page_icon="📄", layout="wide")

# Custom Styling
st.markdown("""
<style>
    /* Dark Theme Setup & Typography Contrast */
    .stApp {
        background-color: #0E1117;
        color: #F8FAFC;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
    }
    p, span, label, li {
        color: #E2E8F0;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1E1E2E !important;
    }
    section[data-testid="stSidebar"] * {
        color: #F8FAFC;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #4C4CFF !important;
        color: #FFFFFF !important;
        border: 1px solid #6366F1 !important;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #3B3BCC !important;
        color: #FFFFFF !important;
        border-color: #4C4CFF !important;
    }

    /* Cards */
    .css-card {
        background-color: #1E1E2E;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 10px;
        border: 1px solid #334155;
    }
    .css-card h4 {
        color: #FFFFFF !important;
        margin-top: 0;
        margin-bottom: 8px;
    }
    .css-card p {
        color: #CBD5E1 !important;
        margin-bottom: 0;
    }

    /* File Uploader & Widgets */
    div[data-testid="stFileUploader"] {
        background-color: #1E1E2E;
        border-radius: 8px;
        padding: 10px;
    }
    div[data-testid="stFileUploader"] * {
        color: #F8FAFC !important;
    }
    div[data-testid="stExpander"] {
        background-color: #1E1E2E;
        border: 1px solid #334155;
        border-radius: 8px;
        color: #F8FAFC;
    }
    input, textarea {
        color: #F8FAFC !important;
        background-color: #1E1E2E !important;
    }
</style>
""", unsafe_allow_html=True)

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "current_pdf_name" not in st.session_state:
    st.session_state.current_pdf_name = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "page_count" not in st.session_state:
    st.session_state.page_count = 0
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

# Helper Functions
def extract_response_text(response):
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)

def clear_session():
    if st.session_state.vector_db is not None:
        try:
            st.session_state.vector_db.delete_collection()
        except Exception:
            pass
    st.session_state.messages = []
    st.session_state.vector_db = None
    st.session_state.retriever = None
    st.session_state.current_pdf_name = None
    st.session_state.pdf_processed = False
    st.session_state.page_count = 0
    st.session_state.chunk_count = 0
    if os.path.exists(CHROMA_DIR):
        try:
            shutil.rmtree(CHROMA_DIR, ignore_errors=True)
        except Exception:
            pass

# Sidebar
with st.sidebar:
    st.title("📄 PDFMind")
    st.markdown("AI-powered PDF Question Answering")
    st.markdown("---")
    
    if st.button("🗑 Start New Document"):
        clear_session()
        st.rerun()

    if not st.session_state.pdf_processed:
        st.markdown("### Upload PDF")
        uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"], label_visibility="collapsed")
        
        if uploaded_file:
            try:
                pdf_reader = pypdf.PdfReader(uploaded_file)
                num_pages = len(pdf_reader.pages)
                st.success(f"✓ PDF uploaded\n\nPages: {num_pages}")
                
                if num_pages > 6:
                    st.error("This PDF contains more than 6 pages. Please upload a shorter PDF.")
                else:
                    if st.button("Process PDF"):
                        if not api_key:
                            st.error("Groq API key is not configured. Please add GROQ_API_KEY to your .env file.")
                        else:
                            with st.status("Processing your PDF...", expanded=True) as status:
                                st.write("📖 Reading PDF...")
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                                    tmp_file.write(uploaded_file.getvalue())
                                    tmp_file_path = tmp_file.name
                                
                                try:
                                    loader = PyPDFLoader(tmp_file_path)
                                    docs = loader.load()
                                    
                                    if not docs:
                                        st.error("No readable text could be extracted from this PDF.")
                                        status.update(label="Extraction failed", state="error")
                                    else:
                                        for doc in docs:
                                            doc.metadata["source"] = uploaded_file.name
                                            if "page" in doc.metadata:
                                                doc.metadata["page"] += 1
                                                
                                        st.write("✂️ Splitting text into chunks...")
                                        text_splitter = RecursiveCharacterTextSplitter(
                                            chunk_size=1000,
                                            chunk_overlap=150
                                        )
                                        chunks = text_splitter.split_documents(docs)
                                        st.write(f"Created {len(chunks)} chunks.")
                                        
                                        st.write("🧠 Creating embeddings...")
                                        st.write("💾 Storing vectors in ChromaDB...")
                                        
                                        # Clear old vector store if present
                                        if os.path.exists(CHROMA_DIR):
                                            try:
                                                shutil.rmtree(CHROMA_DIR, ignore_errors=True)
                                            except Exception:
                                                pass
                                        
                                        embedding_model = HuggingFaceEmbeddings(
                                            model_name="all-MiniLM-L6-v2"
                                        )
                                        
                                        vector_db = Chroma.from_documents(
                                            documents=chunks,
                                            embedding=embedding_model,
                                            persist_directory=CHROMA_DIR
                                        )
                                        
                                        retriever = vector_db.as_retriever(
                                            search_type="mmr",
                                            search_kwargs={
                                                "k": 4,
                                                "fetch_k": 10
                                            }
                                        )
                                        
                                        st.session_state.vector_db = vector_db
                                        st.session_state.retriever = retriever
                                        st.session_state.pdf_processed = True
                                        st.session_state.current_pdf_name = uploaded_file.name
                                        st.session_state.page_count = num_pages
                                        st.session_state.chunk_count = len(chunks)
                                        
                                        status.update(label="✅ PDF ready!", state="complete", expanded=False)
                                        st.rerun()
                                except Exception as e:
                                    import traceback
                                    st.error(f"Error processing PDF: {str(e)}\n\n```python\n{traceback.format_exc()}\n```")
                                    status.update(label="Processing failed", state="error")
                                finally:
                                    if os.path.exists(tmp_file_path):
                                        try:
                                            os.unlink(tmp_file_path)
                                        except Exception:
                                            pass
            except Exception as e:
                st.error("Invalid PDF file.")
    else:
        st.success("✓ PDF processed")
        st.markdown(f"**File:**\n{st.session_state.current_pdf_name}")
        st.markdown(f"**Pages:**\n{st.session_state.page_count}")
        st.markdown(f"**Chunks:**\n{st.session_state.chunk_count}")

    st.markdown("---")
    st.markdown("### Developer Options")
    show_debug = st.checkbox("Show retrieval details")

# Main UI
if not st.session_state.pdf_processed:
    st.markdown("<h1 style='text-align: center; color: #4C4CFF;'>PDFMind</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #CBD5E1; font-size: 1.2rem;'>Understand your documents with AI.</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #CBD5E1;'>Your AI-powered PDF assistant</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("### 📄 Upload your PDF")
    st.markdown("Upload a PDF of up to 6 pages and ask questions about its contents.")
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="css-card">
        <h4>📄 PDF Understanding</h4>
        <p>Extract and analyze text from your PDF documents accurately.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="css-card">
        <h4>🚀 Groq AI</h4>
        <p>Powered by ultra-fast LLMs on Groq's LPU inference engine.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="css-card">
        <h4>🧠 Semantic Search</h4>
        <p>Find exactly what you need quickly with ChromaDB vector search.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="css-card">
        <h4>🔍 Source-based Answers</h4>
        <p>Every answer is strictly grounded in your uploaded document context.</p>
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("### 📄 Current Document")
    st.markdown(f"**{st.session_state.current_pdf_name}**")
    st.markdown("Your PDF is ready. Ask anything about it.")
    st.markdown("---")
    
    # Chat Interface
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(msg["sources"]):
                        st.markdown(f"**Source {i+1}** (Page: {source['page']})")
                        st.markdown(f"> {source['text']}")
            if show_debug and "debug" in msg:
                with st.expander("🛠 Debug Info"):
                    st.write(f"Retrieved chunks: {msg['debug']['chunks_retrieved']}")
                    st.write(f"Retrieved pages: {', '.join(map(str, msg['debug']['pages']))}")

    question = st.chat_input("Ask something about your PDF...")
    
    if question:
        if not api_key:
            st.error("Groq API key is not configured. Please add GROQ_API_KEY to your .env file.")
        else:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
                
            with st.chat_message("assistant"):
                with st.spinner("Searching your PDF and generating an answer..."):
                    try:
                        retriever = st.session_state.retriever
                        retrieved_docs = retriever.invoke(question)
                        
                        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
                        
                        llm = ChatGroq(
                            model_name=GROQ_MODEL,
                            groq_api_key=api_key
                        )
                        
                        prompt = f"""You are PDFMind, a helpful PDF question-answering assistant.

Answer the user's question using ONLY the information
provided in the PDF context below.

Do not use outside knowledge.

Do not make up information.

If the answer is not available in the provided context,
respond exactly with:

"I could not find the answer in the PDF."

Give clear and concise answers.

PDF CONTEXT:
{context}

USER QUESTION:
{question}"""

                        response = llm.invoke(prompt)
                        answer = extract_response_text(response)
                        
                        st.markdown(answer)
                        
                        sources = []
                        pages = []
                        for doc in retrieved_docs:
                            page = doc.metadata.get('page', 'Unknown')
                            text = doc.page_content
                            sources.append({"page": page, "text": text})
                            if page not in pages:
                                pages.append(page)
                                
                        if sources:
                            with st.expander("📚 View Sources"):
                                for i, source in enumerate(sources):
                                    st.markdown(f"**Source {i+1}** (Page: {source['page']})")
                                    st.markdown(f"> {source['text']}")
                                    
                        debug_info = {
                            "chunks_retrieved": len(retrieved_docs),
                            "pages": pages
                        }
                        if show_debug:
                            with st.expander("🛠 Debug Info"):
                                st.write(f"Retrieved chunks: {debug_info['chunks_retrieved']}")
                                st.write(f"Retrieved pages: {', '.join(map(str, debug_info['pages']))}")
                                
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                            "debug": debug_info
                        })
                    except Exception as e:
                        import traceback
                        st.error(f"An error occurred: {str(e)}\n\n```python\n{traceback.format_exc()}\n```")
