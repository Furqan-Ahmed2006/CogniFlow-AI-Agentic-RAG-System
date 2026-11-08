# 🧠 CogniRAG-AI: Enterprise Multilingual RAG & Agentic Workflow Engine

![Python 3.12](https://img.shields.io/badge/Python-3.12%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![LangChain](https://img.shields.io/badge/LangChain-Agentic-green.svg)
![Ragas](https://img.shields.io/badge/Ragas-Benchmarked-yellow.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)

**CogniRAG-AI** is a production-grade, multilingual Retrieval-Augmented Generation (RAG) and Agentic Workflow Engine designed to process complex technical documents, multilingual scripts (Urdu, Roman Urdu, and English), and enterprise agreements.

It combines intelligent document retrieval, agentic intent routing, automated web search fallbacks, and evaluation-driven guardrails to deliver reliable, context-aware, and grounded responses.

---

## **🌟 Key Features**

- **Multilingual & Multi-Script Support:** Seamlessly processes and retrieves information from English, Native Urdu (Nastaliq script), and Roman Urdu documents, including PDF and DOCX files.

- **Agentic Intent Routing & Web Fallback:** Automatically evaluates local retrieval confidence. When relevant information is unavailable in internal documents, it dynamically triggers web search using the Tavily API.

- **Isolated Vector Store Architecture:** Implements strict `thread_id`-based session isolation in ChromaDB to prevent cross-contamination between different document sessions and users.

- **Ragas Benchmarking & Evaluation:** Evaluates system performance using enterprise-oriented metrics, including Faithfulness, Answer Relevancy, Context Precision, and Context Recall.

- **Grounded Response Generation:** Uses retrieved document context to generate evidence-based responses and reduce unsupported answers.

- **Agentic Workflow Management:** Leverages LangGraph for structured state management, conditional routing, and dynamic workflow execution.

---

## **🏗️ System Architecture & Workflow**

CogniRAG-AI follows an agentic RAG architecture that intelligently determines whether a query can be answered using internal documents or requires external web search.

```text
              [ User Query / Document Upload ]
                           |
                           v
              [ Ingestion & Semantic Chunking ]
                           |
                           v
              [ Isolated ChromaDB (thread_id) ]
                           |
                           v
               [ Agentic Intent Router ]
                     (LangGraph)
                           |
                           v
                [ Local RAG Retrieval ]
                           |
                  +--------+--------+
                  |                 |
                  v                 v
             [Relevant]         [No Match]
                  |                 |
                  v                 v
          [Grounded Response]  [Tavily Web Search]
                  |                 |
                  |                 v
                  |          [External Context]
                  |                 |
                  +--------+--------+
                           |
                           v
                [ Response Generation ]
                           |
                           v
               [ Ragas Evaluation Suite ]
                           |
                           v
                [ Streamlit Enterprise UI ]
```

---

## **🛠️ Technology Stack**

| Component | Technology |
|---|---|
| Programming Language | Python 3.12+ |
| Frontend / UI | Streamlit |
| Orchestration | LangChain |
| Agentic Workflows | LangGraph |
| LLM Provider | Groq API |
| Primary LLM | `openai/gpt-oss-120b` |
| Evaluation LLM | `openai/gpt-oss-20b` |
| Vector Database | ChromaDB |
| Embedding Model | `paraphrase-multilingual-MiniLM-L12-v2` |
| Embedding Framework | HuggingFace Sentence Transformers |
| Web Search | Tavily API |
| Evaluation Framework | Ragas |
| Session Isolation | Thread-based ChromaDB Architecture |

---

## **📊 Evaluation Results (Ragas Benchmark)**

CogniRAG-AI was evaluated using multilingual datasets containing employee handbooks, technical specifications, and financial contracts.

| Evaluation Metric | Score |
|---|---:|
| Faithfulness | **1.0000** |
| Context Precision | **1.0000** |
| Context Recall | **1.0000** |
| Answer Relevancy | **0.8532** |

**Evaluation Highlights:**

- **Faithfulness (1.0000):** All evaluated responses were consistent with their retrieved contexts.
- **Context Precision (1.0000):** Retrieved contexts were relevant to the evaluated queries.
- **Context Recall (1.0000):** Relevant information was successfully retrieved in the evaluation dataset.
- **Answer Relevancy (0.8532):** Responses demonstrated strong relevance to user queries.

*Note: These scores represent results on the tested evaluation dataset and do not guarantee zero hallucinations or perfect retrieval in every production scenario.*

---

## **🚀 Live Demo**

**Streamlit Cloud Application:** [Access CogniRAG-AI](https://cogniflow-ai-agentic-rag-system-ctkyfqwk4ypwzt8a9tymfz.streamlit.app/)

*Replace the placeholder link with your deployed Streamlit application URL.*

---

## **⚙️ Local Installation & Setup**

Follow these steps to run CogniRAG-AI locally.

### **1. Clone the Repository**

```bash
git clone https://github.com/YOUR_USERNAME/cognirag-ai.git
cd cognirag-ai
```

### **2. Create and Activate a Virtual Environment**

```bash
python -m venv myenv
```

**Windows:**

```bash
myenv\Scripts\activate
```

**Linux / macOS:**

```bash
source myenv/bin/activate
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

Install the additional Windows-specific file-locking dependency:

```bash
pip install "portalocker[win32]"
```

### **4. Configure Environment Secrets**

Create the following file:

```text
.streamlit/secrets.toml
```

Add your API credentials:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
TAVILY_API_KEY = "your_tavily_api_key_here"
ENVIRONMENT = "production"
```

**Important:** Never commit your API keys or secrets to a public GitHub repository.

### **5. Run the Application**

```bash
streamlit run app.py
```

The application will start locally and provide a URL in your terminal.

---

## **📁 Project Highlights**

- Multilingual document ingestion and processing.
- Semantic chunking and vector-based retrieval.
- Thread-level session isolation.
- Intelligent retrieval confidence evaluation.
- Conditional agentic routing using LangGraph.
- Automated external knowledge retrieval using Tavily.
- Context-grounded response generation.
- Ragas-powered evaluation and benchmarking.
- Interactive Streamlit interface.

---

## **🔐 Security & Reliability Considerations**

CogniRAG-AI incorporates architectural measures to improve reliability and session isolation:

- Thread-based document session separation.
- Environment-based API key management.
- Context-grounded response generation.
- Retrieval-based routing decisions.
- Evaluation-driven quality monitoring.

---

## **👨‍💻 Author**

**Muhammad Furqan Ahmad**


---

## **📄 License**

This project is licensed under the **MIT License**. You are free to use, modify, and distribute this project under the terms of the license.

---

**Built with Python, LangChain, LangGraph, RAG, and Agentic AI.**