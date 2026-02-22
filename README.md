# Gemini Clone

A chat app powered by Groq + LangGraph + Tavily, with a Streamlit UI.

## Features

- **Tavily Search**: Real-time web search for current events, stock prices, news
- **Agentic Routing**: AI decides when to search vs answer directly
- **Streaming Responses**: Zero waiting feel with token-by-token streaming

## Local Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your API keys:
   ```
   GROQ_API_KEY=your_groq_key_here
   TAVILY_API_KEY=your_tavily_key_here
   ```
   - Groq: [console.groq.com](https://console.groq.com/)
   - Tavily: [tavily.com](https://tavily.com/)

3. Run:
   ```bash
   streamlit run app.py
   ```

## Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub and click **New app**
4. Select your repo, branch, and set **Main file path** to `app.py`
5. Add secrets:
   - **GROQ_API_KEY** = your Groq API key
   - **TAVILY_API_KEY** = your Tavily API key
6. Click **Deploy**
