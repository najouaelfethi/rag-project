from rag.rag_engine import RAGEngine

if __name__ == "__main__":
    rag = RAGEngine(
        index_name="test-index",  
        namespace="pdf-multimodal" 
    )

    question = "Can you explain the architecture of the Transformer as shown in the diagram?"
    #question = "What does the multi-head attention diagram in Transformers show?"
    answer = rag.answer_question(question)

    print("\nQuestion:", question)
    print("RAG Answer:\n", answer)
