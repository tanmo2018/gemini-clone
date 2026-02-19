"""LangGraph & Groq orchestration for the Gemini clone chat application."""

import os
from typing import Sequence

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.graph.message import add_messages

load_dotenv()


class ChatState(MessagesState):
    """Chat state extending MessagesState with message history."""
    pass


def create_chat_model():
    """Create and return the Groq chat model."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError(
            "GROQ_API_KEY not set. Add your key to .env (get one at https://console.groq.com/)"
        )
    return ChatGroq(
        api_key=api_key,
        model="llama-3.3-70b-versatile",
        temperature=0.7,
    )


def create_prompt():
    """Create the chat prompt with history placeholder."""
    return ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Be concise, accurate, and friendly."),
        MessagesPlaceholder(variable_name="messages"),
    ])


def chat_node(state: ChatState) -> dict:
    """Process user message and return AI response."""
    model = create_chat_model()
    prompt = create_prompt()
    chain = prompt | model
    response = chain.invoke({"messages": state["messages"]})
    return {"messages": [response]}


def build_graph() -> StateGraph:
    """Build and compile the LangGraph chat workflow."""
    graph = StateGraph(ChatState)

    graph.add_node("chat", chat_node)
    graph.add_edge("chat", END)
    graph.set_entry_point("chat")

    return graph.compile()


def invoke_chat(messages: Sequence[BaseMessage]) -> str:
    """Invoke the chat graph and return the AI response text."""
    compiled = build_graph()
    result = compiled.invoke({"messages": list(messages)})
    if result.get("messages"):
        last_message = result["messages"][-1]
        if isinstance(last_message, AIMessage):
            return last_message.content
        return str(last_message)
    return ""
