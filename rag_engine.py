import os
import chromadb
from pypdf import PdfReader
from docx import Document as DocxDocument
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq
from langchain_core.documents import Document
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

class RAGEngine:
    def __init__(self, thread_id: str = "default_thread"):
        self.thread_id = thread_id
        self.collection_name = f"cognirag_{thread_id}"
        
        self.embeddings = get_embeddings()
        self.tavily_search = TavilySearch(max_results=3)
        
        self.llm = ChatGroq(
            model="openai/gpt-oss-120b", 
            temperature=0.1,
            max_retries=3,
            timeout=30
        )
        
        is_production = False
        try:
            if os.getenv("ENVIRONMENT") == "production":
                is_production = True
            elif hasattr(st, "secrets") and "ENVIRONMENT" in st.secrets:
                if st.secrets["ENVIRONMENT"] == "production":
                    is_production = True
        except Exception:
            is_production = False

        if is_production:
            self.vector_store = Chroma(
                client=chromadb.EphemeralClient(),
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
            )
        else:
            self.persist_directory = "./chroma_db"
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
            )

    def _extract_pdf(self, path: str):
        reader = PdfReader(path)
        docs = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                docs.append(Document(page_content=text, metadata={"source": os.path.basename(path), "page": i + 1}))
        return docs

    def _extract_docx(self, path: str):
        doc = DocxDocument(path)
        full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        if not full_text.strip():
            return []
        return [Document(page_content=full_text, metadata={"source": os.path.basename(path), "page": 1})]

    def process_documents(self, file_paths: list) -> int:
        all_chunks = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        for file_path in file_paths:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".pdf":
                documents = self._extract_pdf(file_path)
            elif ext == ".docx":
                documents = self._extract_docx(file_path)
            else:
                continue

            if documents:
                chunks = splitter.split_documents(documents)
                all_chunks.extend(chunks)

        if all_chunks:
            self.vector_store.add_documents(all_chunks)
        return len(all_chunks)

    def has_documents(self) -> bool:
        try:
            return self.vector_store._collection.count() > 0
        except Exception:
            return False

    def retrieve_docs(self, query: str):
        if not self.has_documents():
            return [], []
        
        docs = self.vector_store.similarity_search(query, k=4)
        scores = [1.0] * len(docs)
        return docs, scores

    def web_fallback(self, query: str):
        try:
            results = self.tavily_search.invoke(query)
            if not results:
                return None
                
            clean_text = ""
            if isinstance(results, list):
                for item in results:
                    if isinstance(item, dict) and "content" in item:
                        clean_text += item["content"] + "\n"
            elif isinstance(results, dict) and "content" in results:
                clean_text = results["content"]
            else:
                clean_text = str(results)
                
            return clean_text if clean_text.strip() else None
        except Exception as e:
            print(f"Tavily Error: {e}")
            return None