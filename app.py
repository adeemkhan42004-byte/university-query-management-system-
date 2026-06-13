import streamlit as st
import pandas as pd
import numpy as np
import faiss
import os
import time
from sentence_transformers import SentenceTransformer
from groq import Groq

st.set_page_config(
    page_title="University Query Management System",
    page_icon="🎓",
    layout="wide"
)

if "bg_color" not in st.session_state:
    st.session_state.bg_color = "#0E1117"

with st.sidebar:
    st.header("Settings")
    bg_choice = st.selectbox(
        "Background Theme",
        ["Dark", "Navy", "Charcoal", "Deep Purple", "Forest"]
    )
    bg_map = {
        "Dark": "#0E1117",
        "Navy": "#0A2342",
        "Charcoal": "#1E1E1E",
        "Deep Purple": "#1A0E2E",
        "Forest": "#0E2A1E"
    }
    st.session_state.bg_color = bg_map[bg_choice]
    st.markdown("---")
    st.caption("University Query Management System")
    st.caption("Built with RAG + Groq LLM")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

st.markdown(f"""
<style>
    .stApp {{
        background-color: {st.session_state.bg_color};
    }}
    .main-title {{
        text-align: center;
        color: #00C9A7;
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 0px;
    }}
    .sub-title {{
        text-align: center;
        color: #94A3B8;
        font-size: 14px;
        margin-top: 0px;
    }}
    .stMarkdown, .stText, p, span, div {{
        color: #FFFFFF !important;
    }}
    .stChatMessage p {{
        color: #FFFFFF !important;
    }}
    .stChatMessage {{
        color: #FFFFFF !important;
    }}
    [data-testid="stChatMessageContent"] p {{
        color: #FFFFFF !important;
    }}
    .stCaption {{
        color: #94A3B8 !important;
    }}
    label, .stSelectbox label {{
        color: #FFFFFF !important;
    }}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">University Query Management System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Ask anything about hostel, mess, fees, placements, subjects, anti-ragging and more</p>', unsafe_allow_html=True)
st.markdown("---")

@st.cache_resource
def load_rag():
    api_key = os.environ.get("Place_your_Groq_API_Key_here", "")

    client = Groq(api_key=api_key)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    csv_files = [
        "students.csv", "hostel.csv", "mess.csv", "anti_ragging.csv",
        "academic_info.csv", "exams.csv", "faculty.csv", "fees.csv",
        "admissions.csv", "scholarships.csv", "library.csv", "clubs.csv",
        "transport.csv", "placements.csv", "grievance.csv", "health.csv",
        "rules.csv", "labs.csv", "contacts.csv", "subjects.csv"
    ]

    documents = []
    sources = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            source_name = file.replace(".csv", "")
            for _, row in df.iterrows():
                text = " | ".join([f"{col}: {val}" for col, val in row.items()])
                documents.append(text)
                sources.append(source_name)
        except FileNotFoundError:
            pass

    embeddings = embedding_model.encode(documents)
    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return client, embedding_model, index, documents, sources

client, embedding_model, index, documents, sources = load_rag()

if "messages" not in st.session_state:
    st.session_state.messages = []

def query_rag(question, chat_history):
    translate_prompt = f"""If the following question is in Hindi, translate it to English.
If it is already in English, return it as is. Return only the question, nothing else.
Question: {question}"""
    translation = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": translate_prompt}]
    )
    search_question = translation.choices[0].message.content.strip()

    question_embedding = embedding_model.encode([search_question])
    question_embedding = np.array(question_embedding).astype("float32")
    distances, indices = index.search(question_embedding, 15)

    retrieved_docs = [documents[i] for i in indices[0]]
    retrieved_sources = [sources[i] for i in indices[0]]
    unique_sources = list(set(retrieved_sources))
    context = "\n".join(retrieved_docs)

    best_distance = float(distances[0][0])
    relevance = max(0, min(100, int(100 - (best_distance * 8))))

    history_text = ""
    recent = chat_history[-12:]
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    prompt = f"""You are a helpful university query assistant for an engineering college.
Use ONLY the context provided below to answer the question accurately.
The context contains real university data — trust it completely.

Previous conversation (use this to understand follow-up questions):
{history_text}

Context:
{context}

Current Question: {question}

Answering rules:
- Use the previous conversation to understand follow-up questions and pronouns like it, that, same
- If the user specifies a format like bullet points, numbered points, table or single line — follow it exactly
- If the user specifies a number of points like 3 or 4 — give exactly that many
- If the user asks in Hindi — answer in Hindi
- If no format is specified — give a clear factual answer in 2 to 3 sentences
- Never say information is not available if it exists in the context above
- Always give the actual data values — phone numbers, timings, names etc

End with:
Source: {", ".join(unique_sources)}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content, relevance

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "relevance" in msg:
            st.caption(f"Relevance Score: {msg['relevance']}%")

if prompt_text := st.chat_input("Type your question here... (or use your keyboard mic button for voice input)"):
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    with st.chat_message("user"):
        st.markdown(prompt_text)

    with st.chat_message("assistant"):
        with st.spinner("Finding answer..."):
            answer, relevance = query_rag(prompt_text, st.session_state.messages)

        placeholder = st.empty()
        displayed = ""
        for word in answer.split(" "):
            displayed += word + " "
            placeholder.markdown(displayed)
            time.sleep(0.03)

        st.caption(f"Relevance Score: {relevance}%")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "relevance": relevance
    })
