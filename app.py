"""Streamlit UI for the Gemini clone chat application."""

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from graph_logic import invoke_chat

st.set_page_config(page_title="Gemini Clone", page_icon="✨", layout="centered")

st.title("✨ Gemini Clone")
st.caption("Powered by Groq + LangGraph")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Convert to LangChain message format
            lc_messages = []
            for m in st.session_state.messages:
                if m["role"] == "user":
                    lc_messages.append(HumanMessage(content=m["content"]))
                else:
                    from langchain_core.messages import AIMessage
                    lc_messages.append(AIMessage(content=m["content"]))

            response = invoke_chat(lc_messages)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
