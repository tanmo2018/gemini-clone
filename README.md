# Gemini Clone

A chat app powered by Groq + LangGraph, with a Streamlit UI.

## Local Setup

1. Clone and install:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your Groq API key:
   ```
   GROQ_API_KEY=your_key_here
   ```
   Get a key at [console.groq.com](https://console.groq.com/)

3. Run:
   ```bash
   streamlit run app.py
   ```

## Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub and click **New app**
4. Select your repo, branch, and set **Main file path** to `app.py`
5. Add secret: **GROQ_API_KEY** = your Groq API key
6. Click **Deploy**
