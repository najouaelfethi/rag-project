
import os
import pytest
from loguru import logger
from core.chunking import DocumentChunker
from core.embedding import EmbeddingGenerator
from core.storage import VectorStore
from config.settings import openai_key, openai_org, pinecone_key, pinecone_index_name

# Test file path
TEST_PDF = "./documents/attention_pdf.pdf"

@pytest.fixture
def chunker():
    return DocumentChunker()

@pytest.fixture
def embedding_generator():
    return EmbeddingGenerator()

@pytest.fixture
def vector_store():
    return VectorStore()

def test_chunking(chunker):
    try:
        #Test with PDF
        chunks = chunker.chunk_document(TEST_PDF)
        
        #Verify chunks
        assert len(chunks) > 0, "No chunks were created"
        assert all("content" in chunk for chunk in chunks), "Missing content in chunks"
        assert all("metadata" in chunk for chunk in chunks), "Missing metadata in chunks"
        
        logger.info(f"Successfully created {len(chunks)} chunks")
        
    except Exception as e:
        logger.error(f"Error in chunking test: {str(e)}")
        raise

def test_embeddings(embedding_generator, chunker):
    try:
        #Get chunks
        chunks = chunker.chunk_document(TEST_PDF)
        
        #Generate embeddings
        chunks_with_embeddings = embedding_generator.generate_embeddings(chunks)
        
        #Verify embeddings
        assert all("embedding" in chunk for chunk in chunks_with_embeddings), "Missing embeddings"
        assert all(len(chunk["embedding"]) > 0 for chunk in chunks_with_embeddings), "Empty embeddings"
        
        logger.info("Successfully generated embeddings")
        
    except Exception as e:
        logger.error(f"Error in embeddings test: {str(e)}")
        raise

def test_storage(vector_store, embedding_generator, chunker):
    """Test vector storage."""
    try:
        #Get chunks with embeddings
        chunks = chunker.chunk_document(TEST_PDF)
        chunks_with_embeddings = embedding_generator.generate_embeddings(chunks)
        
        #Store chunks
        vector_store.store_chunks(chunks_with_embeddings)
        
        #Test search
        query = "What is the main topic of the document?"
        results = vector_store.search_similar(query)
        
        #Verify results
        assert len(results) > 0, "No search results found"
        assert all("content" in result for result in results), "Missing content in results"
        assert all("similarity" in result for result in results), "Missing similarity scores"
        
        logger.info(f"Successfully stored and retrieved {len(results)} chunks")
        
    except Exception as e:
        logger.error(f"Error in storage test: {str(e)}")
        raise

def test_full_rag_system():
    try:
        #Initialize components
        chunker = DocumentChunker()
        embedding_generator = EmbeddingGenerator()
        vector_store = VectorStore()
        
        #Process document
        chunks = chunker.chunk_document(TEST_PDF)
        chunks_with_embeddings = embedding_generator.generate_embeddings(chunks)
        vector_store.store_chunks(chunks_with_embeddings)
        
        #Test query
        query = "What is the main topic of the document?"
        results = vector_store.search_similar(query)
        
        #Verify results
        assert len(results) > 0, "No results found"
        assert all(isinstance(result["similarity"], float) for result in results), "Invalid similarity scores"
        
        logger.info("RAG system test completed successfully")
        
    except Exception as e:
        logger.error(f"Error in RAG system test: {str(e)}")
        raise

if __name__ == "__main__":
    #Configure logger
    logger.add("test_rag.log", rotation="1 MB")
    
    #Run tests
    #test_chunking(DocumentChunker())
    #test_embeddings(EmbeddingGenerator(), DocumentChunker())
    #test_storage(VectorStore(), EmbeddingGenerator(), DocumentChunker())
    #test_full_rag_system()