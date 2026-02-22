"""Streamlit UI for the Gemini clone with streaming and web search."""

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from graph_logic import stream_chat

st.set_page_config(page_title="Gemini Clone", page_icon="✨", layout="centered")

st.title("✨ Gemini Clone")
st.caption("Powered by Groq + LangGraph + Tavily Search")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything... I can search the web too!"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        lc_messages = []
        for m in st.session_state.messages:
            if m["role"] == "user":
                lc_messages.append(HumanMessage(content=m["content"]))
            else:
                lc_messages.append(AIMessage(content=m["content"]))

        response_placeholder = st.empty()
        full_response = ""
        
        for chunk in stream_chat(lc_messages):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")
        
        response_placeholder.markdown(full_response)

    clean_response = full_response.replace("🔍 *Searching the web...*\n\n", "")
    st.session_state.messages.append({"role": "assistant", "content": clean_response})
