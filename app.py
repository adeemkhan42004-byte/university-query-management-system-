import streamlit as st
import pandas as pd
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer
from groq import Groq

st.set_page_config(
    page_title="University Query Management System",
    page_icon="🎓",
    layout="centered"
)

st.title("University Query Management System")
st.markdown("Ask anything about admissions, hostel, mess, fees, placements, anti-ragging and more.")
st.markdown("---")

@st.cache_resource
def load_rag():
    # API key loaded from environment variable for security
    api_key = os.environ.get("GROQ_API_KEY", "")
    client = Groq(api_key=api_key)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    csv_files = [
        "students.csv", "hostel.csv", "mess.csv", "anti_ragging.csv",
        "academic_info.csv", "exams.csv", "faculty.csv", "fees.csv",
        "admissions.csv", "scholarships.csv", "library.csv", "clubs.csv",
        "transport.csv", "placements.csv", "grievance.csv", "health.csv",
        "rules.csv", "labs.csv", "contacts.csv"
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

def query_rag(question, n_results=5):
    question_embedding = embedding_model.encode([question])
    question_embedding = np.array(question_embedding).astype("float32")
    distances, indices = index.search(question_embedding, n_results)
    retrieved_docs = [documents[i] for i in indices[0]]
    retrieved_sources = [sources[i] for i in indices[0]]
    unique_sources = list(set(retrieved_sources))
    context = "\n".join(retrieved_docs)

    prompt = f"""You are a university query assistant.
Answer ONLY based on the context below. Be specific and direct.
If the exact answer is in the context, state it clearly.
Do not say information is not available if it is present in the context.

Context:
{context}

Question: {question}

Give a direct specific answer in 2-3 sentences maximum.
End with:
Source: {", ".join(unique_sources)}"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

st.subheader("Ask a Question")
question = st.text_input("Type your question here:", placeholder="e.g. What is the mess menu for Monday?")

if st.button("Get Answer"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Finding answer..."):
            answer = query_rag(question)
        st.markdown("### Answer")
        st.markdown(answer)

st.markdown("---")
st.markdown("**Sample Questions:**")
st.markdown("- What is the anti-ragging helpline number?")
st.markdown("- What is the mess menu for Monday?")
st.markdown("- Who is the HOD of CSE department?")
st.markdown("- What is the library timing?")
st.markdown("- Which companies came for placements?")
st.markdown("- What is the hostel curfew time?")
st.markdown("- What scholarships are available?")
st.markdown("- What is the fee structure for CSE?")
st.markdown("- How do I report a grievance?")
st.markdown("- What are the transport routes available?")
