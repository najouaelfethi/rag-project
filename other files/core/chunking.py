from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from loguru import logger
from config.settings import model_embeddings

class DocumentChunker:
    def __init__(self):
        self.semantic_splitter = SemanticChunker(model_embeddings)

    def chunk_document(self, file_path: str) -> List[Dict[str, Any]]:
        try:
            #Load document 
            docs = self._load_document(file_path)
            
            #Split document
            if file_path.endswith('.pdf'):
                chunks = self.semantic_splitter.split_documents(docs)
            
            #Format chunks with metadata
            formatted_chunks = []
            for i, chunk in enumerate(chunks):
                formatted_chunks.append({
                    "id": f"chunk_{i}",
                    "content": chunk.page_content,
                    "metadata": {
                        "source": file_path,
                        "page": chunk.metadata.get("page", 0),
                        "type": self._get_file_type(file_path)
                    }
                })
            
            logger.info(f"Created {len(formatted_chunks)} chunks from {file_path}")
            return formatted_chunks
            
        except Exception as e:
            logger.error(f"Error chunking document {file_path}: {str(e)}")
            raise

    def _load_document(self, file_path: str) -> List[Document]:
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)

        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        
        return loader.load()

    def _get_file_type(self, file_path: str) -> str:
        if file_path.endswith('.pdf'):
            return "pdf"

        else:
            return "unknown"