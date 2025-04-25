from langchain.docstore.document import Document
from core.document_loader import DocumentLoader

if __name__ == "__main__":
    test_path = "./documents/attention_pdf.pdf"  # or .docx or .pptx
    try:
        documents = DocumentLoader.load_document(test_path)
        print(f"Loaded {len(documents)} document chunks")
        for i, doc in enumerate(documents[:3]):  # print first 3 chunks only
            print(f"\n--- Chunk {i + 1} ---")
            print(doc.page_content if hasattr(doc, "page_content") else doc)
    except Exception as e:
        print(f"Error: {e}")
