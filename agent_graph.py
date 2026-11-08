from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document

class AgentState(TypedDict):
    query: str
    documents: List[Document]
    generation: str
    used_web: bool

class CogniRAGWorkflow:
    def __init__(self, rag_engine):
        self.engine = rag_engine
        self.workflow = StateGraph(AgentState)
        self._build_graph()

    def _retrieve_node(self, state: AgentState):
        docs, scores = self.engine.retrieve_docs(state["query"])
        return {"documents": docs}

    def _decide_route(self, state: AgentState):
        docs = state.get("documents", [])
        if not docs:
            return "web_search"

        
        context = "\n\n".join([d.page_content for d in docs])
        router_prompt = f"""You are a strict query-context relevance evaluator.
Evaluate if the provided context contains enough information to answer the user's query.

Query: {state['query']}
Context: {context}

Reply with ONLY 'YES' if the context contains relevant information to answer the query.
Reply with ONLY 'NO' if the context does NOT contain relevant information or is unrelated.

Evaluation:"""

        try:
            decision = self.engine.llm.invoke(router_prompt).content.strip().upper()
            if "YES" in decision:
                return "generate"
        except Exception:
            pass

        return "web_search"

    def _web_search_node(self, state: AgentState):
        clean_web_text = self.engine.web_fallback(state["query"])
        
        if not clean_web_text:
            return {"documents": [], "used_web": True}
        
        web_doc = Document(page_content=clean_web_text, metadata={"source": "Tavily Web", "page": 1})
        return {"documents": [web_doc], "used_web": True}

    def _generate_node(self, state: AgentState):
        query = state["query"]
        docs = state.get("documents", [])

        if not docs:
            return {"generation": "Unable to find any relevant information."}

        context = "\n\n".join([d.page_content for d in docs])
        
        prompt = f"""You are CogniRAG-AI, a strict script-matching assistant.
Answer the user's query using ONLY the provided context.

SCRIPT MATCHING RULE (STRICT):
- User input script MUST match Output script exactly.
- If query is in Latin script (e.g. "lahore ka temperature kia hai?" or "Annual leave kitni hai?"), respond strictly in ROMAN URDU (Latin script). DO NOT use Arabic/Urdu script (like اردو).
- If query is in Urdu script (e.g. "انول لیو کتنی hai"), respond in Urdu script.
- If query is in English, respond in English.

Context:
{context}

Query: {query}

Answer:"""
        try:
            response = self.engine.llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            answer = f"Error generating response: {e}"

        return {"generation": answer}

    def _build_graph(self):
        self.workflow.add_node("retrieve", self._retrieve_node)
        self.workflow.add_node("web_search", self._web_search_node)
        self.workflow.add_node("generate", self._generate_node)

        self.workflow.set_entry_point("retrieve")
        
        self.workflow.add_conditional_edges(
            "retrieve",
            self._decide_route,
            {"web_search": "web_search", "generate": "generate"}
        )
        self.workflow.add_edge("web_search", "generate")
        self.workflow.add_edge("generate", END)
        self.app = self.workflow.compile()

    def stream_response(self, query: str):
        state = self.app.invoke({
            "query": query, "documents": [], "generation": "", "used_web": False
        })
        
        prompt = state["generation"]
        
        if prompt.startswith("Unable to find"):
            yield prompt
            return
            
        for chunk in self.engine.llm.stream(prompt):
            if chunk.content:
                yield chunk.content