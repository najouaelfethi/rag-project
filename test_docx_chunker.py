# test_docx_chunking.py
from core.chunker import DocumentChunker
from core.document_loader import DocumentLoader
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

def test_docx_chunking():
    # Initialize components
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_organization=os.getenv("OPENAI_ORG_ID")
    )
    chunker = DocumentChunker(embeddings)
    
    file_path = "./documents/paper Optimization.docx"
    
    try:
        chunks = chunker.chunk_docx(file_path)
        
        print(f"\nTotal chunks: {len(chunks)}")
        
        for i, chunk in enumerate(chunks, 1):
            print(f"\n--- Chunk {i} ---")
            print(f"{chunk.page_content[:200]}...")  # First 200 chars
                  
    except Exception as e:
        print(f"Error during testing: {str(e)}")

if __name__ == "__main__":
    test_docx_chunking()