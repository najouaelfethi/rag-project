#Storage of embeddings + search similarity
from typing import List, Dict, Any
from loguru import logger
from config.settings import index, pinecone_index_name

class VectorStore:
    def __init__(self):
        self.index = index
        self.namespace = pinecone_index_name

    def store_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        try:
            # Prepare data for storage
            vectors = []
            for i, chunk in enumerate(chunks):
                vector = {
                    "id": f"chunk_{i}",
                    "values": chunk["embedding"],
                    "metadata": {
                        "content": chunk["content"],
                        "type": chunk["metadata"]["type"],
                        "source": chunk["metadata"]["source"],
                        "page": chunk["metadata"]["page"]
                    }
                }
                vectors.append(vector)
            
            #Store in Pinecone
            self.index.upsert(
                vectors=vectors,
                namespace=self.namespace
            )
            
            logger.info(f"Stored {len(chunks)} chunks in Pinecone")
            
        except Exception as e:
            logger.error(f"Error storing chunks in Pinecone: {str(e)}")
            raise

    def search_similar(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                namespace=self.namespace
            )
            
            # Format results
            similar_chunks = []
            for match in results.matches:
                similar_chunks.append({
                    "id": match.id,
                    "content": match.metadata["content"],
                    "metadata": {
                        "type": match.metadata["type"],
                        "source": match.metadata["source"],
                        "page": match.metadata["page"]
                    },
                    "similarity": match.score
                })
            
            return similar_chunks
            
        except Exception as e:
            logger.error(f"Error searching similar chunks in Pinecone: {str(e)}")
            raise