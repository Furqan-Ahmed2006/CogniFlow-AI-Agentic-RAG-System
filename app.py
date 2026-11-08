import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from rag_engine import RAGEngine
from guardrails import Guardrails
from agent_graph import CogniRAGWorkflow

load_dotenv()

st.set_page_config(page_title="CogniRAG-AI", page_icon="🤖", layout="wide")

def create_new_chat():
    new_id = str(uuid.uuid4())
    st.session_state.current_thread_id = new_id
    st.session_state.threads[new_id] = {
        "title": "New Chat",
        "messages": [],
        "rag_engine": RAGEngine(new_id),
        "has_doc": False,
        "pinned": False
    }
    st.session_state.workflow = CogniRAGWorkflow(st.session_state.threads[new_id]["rag_engine"])

def switch_chat(tid):
    st.session_state.current_thread_id = tid
    st.session_state.workflow = CogniRAGWorkflow(st.session_state.threads[tid]["rag_engine"])

def delete_chat(tid):
    if tid in st.session_state.threads:
        del st.session_state.threads[tid]
    if st.session_state.current_thread_id == tid:
        create_new_chat()

def toggle_pin(tid):
    st.session_state.threads[tid]["pinned"] = not st.session_state.threads[tid].get("pinned", False)

def rename_chat(tid, new_title):
    if new_title.strip():
        st.session_state.threads[tid]["title"] = new_title.strip()

if "threads" not in st.session_state:
    st.session_state.threads = {}

if "current_thread_id" not in st.session_state:
    create_new_chat()

if st.session_state.current_thread_id not in st.session_state.threads:
    create_new_chat()

current_thread = st.session_state.threads[st.session_state.current_thread_id]
rag_engine = current_thread["rag_engine"]
workflow = st.session_state.workflow


with st.sidebar:
    st.button("➕ New Chat", use_container_width=True, on_click=create_new_chat)
    st.divider()
    

    sorted_threads = sorted(
        st.session_state.threads.items(),
        key=lambda x: (x[1].get("pinned", False), x[1]["title"]),
        reverse=True
    )

    for tid, thread_data in sorted_threads:
        
        is_active = (tid == st.session_state.current_thread_id)
        title_text = f"📌 {thread_data['title']}" if thread_data.get('pinned') else f"💬 {thread_data['title']}"
        if is_active:
            title_text = f"👉 {title_text}"

        col1, col2 = st.columns([4.5, 1.5])

        with col1:
            st.button(
                title_text, 
                key=f"btn_{tid}", 
                use_container_width=True, 
                on_click=switch_chat, 
                args=(tid,)
            )

        with col2:
    
            with st.popover("⋮", use_container_width=True):
                st.markdown("**Options**")
                
        
                new_name = st.text_input("Rename", value=thread_data["title"], key=f"rename_{tid}")
                if new_name != thread_data["title"]:
                    rename_chat(tid, new_name)
                    st.rerun()

                st.divider()
                
    
                pin_label = "📍 Unpin" if thread_data.get("pinned") else "📌 Pin"
                st.button(pin_label, key=f"pin_{tid}", on_click=toggle_pin, args=(tid,), use_container_width=True)
                
                
                st.button("🗑️ Delete", key=f"del_{tid}", on_click=delete_chat, args=(tid,), use_container_width=True, type="primary")

    st.divider()

    st.markdown("### 📁 Document Ingestion")
    uploaded_files = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"], accept_multiple_files=True)
    
    if uploaded_files and st.button("📥 Process Documents", use_container_width=True):
        os.makedirs("uploads", exist_ok=True)
        temp_paths = []
        
        for uploaded_file in uploaded_files:
            temp_path = os.path.join("uploads", uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            temp_paths.append(temp_path)
            
        with st.spinner("Processing documents..."):
            chunks = rag_engine.process_documents(temp_paths)
            
        if chunks > 0:
            st.success(f"✅ Processed {chunks} chunks!")
            current_thread["has_doc"] = True
            if current_thread["title"] == "New Chat":
                current_thread["title"] = uploaded_files[0].name[:20] + "..."
        else:
            st.error("❌ No readable text found in documents.")
            
        for p in temp_paths:
            if os.path.exists(p): os.remove(p)
        st.rerun()

    # Status Check
    if current_thread["has_doc"]:
        st.success("✅ Documents ready for Q&A")
    else:
        st.info("ℹ️ Please upload a document to begin.")


st.title("🛡️ CogniRAG-AI")
st.markdown("Ask questions in **any language** — get instant, accurate answers from your documents.")


for msg in current_thread["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("🔍 Sources"):
                for src in msg["sources"]:
                    st.markdown(f"- {src}")

if query := st.chat_input("Ask in any language..."):

    if not current_thread["has_doc"]:
        st.warning("📄 Please upload a document first before asking questions.")
        st.stop()


    sec = Guardrails.inspect_input(query)
    if not sec.get("is_safe", False):
        st.error(f"❌ Blocked: {sec.get('reason')}")
        st.stop()


    current_thread["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            state = workflow.app.invoke({
                "query": query, "documents": [], "generation": "", "used_web": False
            })
            
        docs = state.get("documents", [])
        used_web = state.get("used_web", False)
        
        if used_web:
            st.info("ℹ️ Answer not found in document — searched the web for you.")
            
        st.markdown("**💡 Answer:**")
        
    
        response_text = st.write_stream(workflow.stream_response(query))
        
        
        sources_list = []
        if docs:
            for d in docs:
                sources_list.append(f"{d.metadata.get('source', 'Web')} (Page {d.metadata.get('page', '')})")
                
        current_thread["messages"].append({
            "role": "assistant", 
            "content": response_text,
            "sources": sources_list
        })
        
        if sources_list:
            with st.expander("🔍 Sources"):
                for src in sources_list:
                    st.markdown(f"- {src}")