"""LangGraph & Groq orchestration with Tavily search and agentic routing."""

import os
from typing import Literal, Generator, Sequence
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, MessagesState, END
from tavily import TavilyClient

load_dotenv()


class AgentState(MessagesState):
    """State with messages and optional search results."""
    search_results: str = ""
    needs_search: bool = False


def get_groq_model(streaming: bool = False):
    """Create Groq chat model."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or "your_groq" in api_key:
        raise ValueError("GROQ_API_KEY not set. Get one at https://console.groq.com/")
    return ChatGroq(
        api_key=api_key,
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        streaming=streaming,
    )


def get_tavily_client():
    """Create Tavily client for web search."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key or "your_tavily" in api_key:
        raise ValueError("TAVILY_API_KEY not set. Get one at https://tavily.com/")
    return TavilyClient(api_key=api_key)


def router_node(state: AgentState) -> dict:
    """Decide if the query needs a web search."""
    model = get_groq_model()
    
    last_message = state["messages"][-1].content if state["messages"] else ""
    
    router_prompt = f"""Analyze this user query and decide if it requires a web search for current/real-time information.

Query: {last_message}

Respond with ONLY one word:
- "search" if the query asks about: current events, today's news, live prices, weather, recent happenings, anything that changes frequently, or facts you're unsure about
- "direct" if you can answer confidently from your knowledge (general knowledge, coding help, explanations, creative tasks)

Your decision:"""

    response = model.invoke([HumanMessage(content=router_prompt)])
    decision = response.content.strip().lower()
    
    needs_search = "search" in decision
    return {"needs_search": needs_search}


def route_decision(state: AgentState) -> Literal["search", "respond"]:
    """Route based on search decision."""
    return "search" if state.get("needs_search", False) else "respond"


def search_node(state: AgentState) -> dict:
    """Perform web search using Tavily."""
    tavily = get_tavily_client()
    
    last_message = state["messages"][-1].content if state["messages"] else ""
    
    try:
        results = tavily.search(query=last_message, max_results=5)
        
        formatted_results = []
        for r in results.get("results", []):
            formatted_results.append(f"**{r.get('title', 'No title')}**\n{r.get('content', '')}\nSource: {r.get('url', '')}")
        
        search_context = "\n\n---\n\n".join(formatted_results) if formatted_results else "No results found."
    except Exception as e:
        search_context = f"Search failed: {str(e)}"
    
    return {"search_results": search_context}


def respond_node(state: AgentState) -> dict:
    """Generate response, optionally using search results."""
    model = get_groq_model()
    
    messages = list(state["messages"])
    search_results = state.get("search_results", "")
    
    if search_results:
        system_msg = SystemMessage(content=f"""You are a helpful AI assistant with access to real-time web search.

Here are the latest search results for the user's query:

{search_results}

Use this information to provide an accurate, up-to-date response. Cite sources when relevant. If the search results don't fully answer the question, say so and provide what you know.""")
        messages = [system_msg] + messages
    else:
        system_msg = SystemMessage(content="You are a helpful AI assistant. Be concise, accurate, and friendly.")
        messages = [system_msg] + messages
    
    response = model.invoke(messages)
    return {"messages": [response]}


def build_graph():
    """Build the agentic graph with routing."""
    graph = StateGraph(AgentState)
    
    graph.add_node("router", router_node)
    graph.add_node("search", search_node)
    graph.add_node("respond", respond_node)
    
    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route_decision, {"search": "search", "respond": "respond"})
    graph.add_edge("search", "respond")
    graph.add_edge("respond", END)
    
    return graph.compile()


def stream_chat(messages: Sequence[BaseMessage]) -> Generator[str, None, None]:
    """Stream chat response with agentic routing."""
    
    model = get_groq_model()
    last_message = messages[-1].content if messages else ""
    
    router_prompt = f"""Analyze this user query and decide if it requires a web search.

Query: {last_message}

Respond with ONLY "search" or "direct":"""

    router_response = model.invoke([HumanMessage(content=router_prompt)])
    needs_search = "search" in router_response.content.strip().lower()
    
    search_results = ""
    if needs_search:
        yield "🔍 *Searching the web...*\n\n"
        try:
            tavily = get_tavily_client()
            results = tavily.search(query=last_message, max_results=5)
            
            formatted = []
            for r in results.get("results", []):
                formatted.append(f"**{r.get('title', '')}**\n{r.get('content', '')}\nSource: {r.get('url', '')}")
            search_results = "\n\n---\n\n".join(formatted) if formatted else ""
        except Exception as e:
            yield f"⚠️ Search failed: {e}\n\n"
    
    if search_results:
        system_msg = SystemMessage(content=f"""You are a helpful AI with real-time web search.

Search results:
{search_results}

Use these results to answer accurately. Cite sources when helpful.""")
        full_messages = [system_msg] + list(messages)
    else:
        system_msg = SystemMessage(content="You are a helpful AI assistant. Be concise, accurate, and friendly.")
        full_messages = [system_msg] + list(messages)
    
    streaming_model = get_groq_model(streaming=True)
    for chunk in streaming_model.stream(full_messages):
        if chunk.content:
            yield chunk.content


def invoke_chat(messages: Sequence[BaseMessage]) -> str:
    """Non-streaming chat invocation (fallback)."""
    compiled = build_graph()
    result = compiled.invoke({"messages": list(messages), "search_results": "", "needs_search": False})
    if result.get("messages"):
        last_message = result["messages"][-1]
        if isinstance(last_message, AIMessage):
            return last_message.content
        return str(last_message)
    return ""
