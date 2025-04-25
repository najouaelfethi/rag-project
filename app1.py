# app.py
import streamlit as st
import tempfile
from core.document_loader import DocumentLoader
from core.chunker import DocumentChunker
from processors.text_processor import TextProcessor
from processors.image_processor import ImageTextProcessor
from storage.pinecone_vector import VectorStore
from rag.rag_engine import RAGEngine
import os

st.set_page_config(page_title="RAG PDF Assistant", page_icon="📄")

st.title("MultiModal RAG Assistant - IDEO FACTORY ")
uploaded_file = st.file_uploader("Téléversez un fichier PDF", type=["pdf"])

if uploaded_file:
    # Sauvegarder temporairement le fichier
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.success("✅ Fichier téléversé avec succès.")
    st.info("⏳ Traitement du fichier...")

    ## ==== INGESTION + EMBEDDING ==== ##
    file_path = tmp_path
    doc_type = DocumentLoader.detect_type(file_path)

    # TEXT
    text_processor = TextProcessor()
    docs = DocumentLoader.load_document(file_path)
    chunker = DocumentChunker(embeddings=text_processor.embeddings)
    chunks = chunker.chunk_document(file_path, doc_type=doc_type)

    text_vectors = []
    for i, chunk in enumerate(chunks):
        text = chunk.page_content if hasattr(chunk, "page_content") else chunk
        processed = text_processor.process_text(text)
        text_vectors.append({
            "id": f"text_{i}",
            "values": processed["embedding"],
            "metadata": {
                "type": "text",
                "content": processed["content"][:100],
                "source": os.path.basename(file_path)
            }
        })

    # IMAGE
    img_processor = ImageTextProcessor()
    image_paths = img_processor.extract_images_from_document(file_path, output_dir="./extracted_images")
    image_results = img_processor.extract_text_from_images(image_paths)

    image_vectors = []
    for i, r in enumerate(image_results):
        image_vectors.append({
            "id": f"image_{i}",
            "values": r["embedding"],
            "metadata": {
                "type": "image",
                "image_path": r["image_path"],
                "content": r["content"][:100],
                "source": os.path.basename(file_path)
            }
        })

    # Pinecone Storage
    namespace = os.path.splitext(os.path.basename(file_path))[0]
    store = VectorStore()
    all_vectors = text_vectors + image_vectors  
    store.store_vectors(all_vectors, namespace=namespace)

    st.success("✅ Fichier indexé avec succès. Vous pouvez poser vos questions ci-dessous !")

    ## ==== QUESTION ==== ##
    rag = RAGEngine(index_name="test-index", namespace=namespace)

    question = st.text_input("Posez votre question sur ce document 📄")

    if question:
        with st.spinner("🔍 Recherche de la réponse..."):
            response, context = rag.answer_question(question, return_sources=True)

        st.subheader("💬 Réponse :")
        st.write(response)

        with st.expander("🧠 Contexte utilisé"):
            for doc in context:
                st.markdown(f"- {doc.page_content[:200]}...")

