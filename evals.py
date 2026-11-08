import sys
import types
dummy_chat = types.ModuleType("langchain_community.chat_models.vertexai")
dummy_chat.ChatVertexAI = type("ChatVertexAI", (object,), {})
sys.modules["langchain_community.chat_models.vertexai"] = dummy_chat

import os
from datasets import Dataset
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

from rag_engine import RAGEngine
from agent_graph import CogniRAGWorkflow

raw_llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0) 
ragas_llm = LangchainLLMWrapper(raw_llm)
raw_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
ragas_embeddings = LangchainEmbeddingsWrapper(raw_embeddings)

def run_ragas_production_evaluation():
    print("==================================================")
    print("🚀 Running Enterprise Ragas Metric Suite (Groq + HF)")
    print("==================================================")

    MASTER_TEST_CASES = [
        {
            "doc": "01_English_Employee_Handbook.pdf",
            "thread": "ragas_doc_1",
            "query": "When was Northstar Digital Services founded, and who founded it?",
            "ground_truth": "Northstar Digital Services was founded on 12 October 2015 by Sara Malik and Hamza Qureshi."
        },
        {
            "doc": "02_Urdu_Employee_Handbook.pdf",
            "thread": "ragas_doc_2",
            "query": "ملازمین کو سالانہ کتنی چھٹیاں ملتی ہیں؟",
            "ground_truth": "18 سالانہ چھٹیاں"
        },
        {
            "doc": "03_Roman_Urdu_HomeHub_S2_User_Manual.docx",
            "thread": "ragas_doc_3",
            "query": "HomeHub S2 ki warranty kitne arsay ki hai?",
            "ground_truth": "24-month limited warranty"
        },
        {
            "doc": "04_English_HomeHub_S2_Technical_Specification.docx",
            "thread": "ragas_doc_4",
            "query": "What is the recommended retail price, and what power adapter is required?",
            "ground_truth": "PKR 8,490"
        },
        {
            "doc": "05_AI_in_Healthcare_Research_Summary.pdf",
            "thread": "ragas_doc_5",
            "query": "Who is the current CEO of OpenAI?",
            "ground_truth": "Sam Altman"
        },
        {
            "doc": "06_Cloud_Data_Platform_Service_Agreement.pdf",
            "thread": "ragas_doc_6",
            "query": "What is the total professional fee, excluding taxes and cloud charges?",
            "ground_truth": "PKR 1,800,000"
        }
    ]

    questions, answers, contexts, ground_truths = [], [], [], []

    print("\n📂 Executing Pipeline over Test Cases...")

    for item in MASTER_TEST_CASES:
        file_path = os.path.join("uploads", item["doc"])
        if not os.path.exists(file_path):
            print(f"⚠️ File not found -> {file_path}")
            continue

        print(f"   -> Extracting context and generation for: {item['doc']}")
        
        engine = RAGEngine(thread_id=item["thread"])
        engine.process_documents([file_path])
        workflow = CogniRAGWorkflow(engine)

        state = workflow.app.invoke({
            "query": item["query"], 
            "documents": [], 
            "generation": "", 
            "used_web": False
        })

        retrieved_docs = state.get("documents", [])
        actual_response = state.get("generation", "")
        retrieved_contexts = [doc.page_content for doc in retrieved_docs] if retrieved_docs else ["No context available"]

        questions.append(item["query"])
        answers.append(actual_response)
        contexts.append(retrieved_contexts)
        ground_truths.append(item["ground_truth"])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    print("\n📊 Computing Ragas Metrics with Groq & HuggingFace...")

    # Instantiate Metric Objects with Groq LLM wrapper
    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
    ]

    
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        embeddings=ragas_embeddings
    )

    print("\n==================================================")
    print("Enterprise Benchmark Complete!")
    print("==================================================")
    print(result)

    df = result.to_pandas()
    df.to_csv("ragas_evaluation_report.csv", index=False)
    print("\n📄 Detailed report saved to 'ragas_evaluation_report.csv'")

if __name__ == "__main__":
    run_ragas_production_evaluation()