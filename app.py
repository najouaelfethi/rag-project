import streamlit as st
from rag.rag_engine import RAGEngine  # Chemin vers ton RAGEngine
from config.settings import Config

# Initialiser le moteur RAG
rag = RAGEngine(index_name="test-index", namespace="pdf-multimodal") 

st.set_page_config(page_title="RAG Multimodal", layout="wide")
st.title("📄Multimodal RAG ASSISTANT - IDEO FACTORY")

# Zone de saisie de la question
question = st.text_input("Ask about document:", placeholder="write here")

if st.button("ASK"):
    if not question.strip():
        st.warning("Please enter a question")
    else:
        with st.spinner("Searching and generating response ..."):
            response, context_docs = rag.answer_question(question, return_sources=True)

        # Afficher la réponse
        st.success("Generated response")
        st.write(response)

        # Afficher les documents utilisés
        st.markdown("---")
        st.markdown("### Chunks used from Pinecone")
        for i, doc in enumerate(context_docs):
            st.markdown(f"**Chunk {i+1}** — *Type: {doc.metadata.get('type', 'unknown')}*")
            st.code(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
