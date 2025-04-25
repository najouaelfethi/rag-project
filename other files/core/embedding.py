from langchain_openai import OpenAIEmbeddings
from typing import List,Dict,Any
from config.settings import model_embeddings


class EmbeddingGenerator:
    def __init__(self):
        self.model = model_embeddings

    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            texts = [chunk["content"] for chunk in chunks]
            embeddings = self.model.embed_documents(texts)
            
            for i, chunk in enumerate(chunks):
                chunk["embedding"] = embeddings[i]
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise