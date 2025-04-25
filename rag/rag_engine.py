from typing import List
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from storage.pinecone_vector import VectorStore
from config.settings import Config
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class RAGEngine:
    def __init__(self, index_name:str, namespace: str):
        self.index_name = index_name
        self.namespace = namespace
        self.store = VectorStore()
        self.llm = ChatOpenAI(openai_api_key=Config.openai_key)

    def retrieve_context(self, query: str, top_k: int = 3) -> List[Document]:
        #Embed the query
        embedding = self.store.embed_query(query)
        
        #Search Pinecone vector store
        results = self.store.search_vectors(query_vector=embedding, namespace=self.namespace, top_k=top_k)

        #Convert Pinecone results into LangChain Documents
        documents = []
        for match in results.matches:
            metadata = match.metadata
            content = metadata.get("content", "")
            documents.append(Document(page_content=content, metadata=metadata))

        return documents

    def answer_question(self, query: str, return_sources=False):
        docs = self.retrieve_context(query)
        context = "\n".join([doc.page_content for doc in docs])

        #prompt = f"Answer the following question from context:\n\n{context}\n\nQuestion: {query}"

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are a helpful assistant. Use the following context to answer the question.

            Context:
            {context}

            Question:
            {question}

            Answer:
            """
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)

        result = chain.invoke({"context": context, "question": query})  # ⬅️ ici .invoke

        if return_sources:
            return result["text"], docs 
        else:
            return result["text"]      

    def embed_query(self, query: str) -> List[float]:
        from langchain_openai import OpenAIEmbeddings
        embedder = OpenAIEmbeddings(openai_api_key=Config.openai_key)
        return embedder.embed_query(query)
