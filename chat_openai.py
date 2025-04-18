import os
import chromadb
import openai
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader,UnstructuredWordDocumentLoader,UnstructuredPowerPointLoader
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter,MarkdownHeaderTextSplitter
import json
from documents_all import load_documents
from langchain_experimental.text_splitter import SemanticChunker
from pinecone import Pinecone, ServerlessSpec
from transformers import pipeline
from langchain.prompts import PromptTemplate

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
openai_org = os.getenv("OPENAI_ORG_ID")
pinecone_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")


pc = Pinecone(api_key=pinecone_key)
index = pc.Index(pinecone_index_name)

chat = ChatOpenAI(temperature=0.3,model_name="gpt-4o-mini",openai_api_key=openai_key,openai_organization=openai_org)
model_embeddings = OpenAIEmbeddings(openai_api_key=openai_key,openai_organization=openai_org)

def prompt_template():
    prompt = PromptTemplate(
        template="""Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know.
        
        Context: {context} 
        
        Question: {question}
        
        Answer:""",
        input_variables=["context", "question"]
    )
    return prompt

def chunk_pdf(file_path): #Splitting using Semantic Chunker => split by meaning
    docs = PyPDFLoader(file_path).load()#load and variable stores content of PDF
    splitter = SemanticChunker(model_embeddings)#split doc into chunks, each chunk is of size 600 and each new chunk repeats 100 characters fro the previous one(chunk)=>AI will remeber context better
    chunks = splitter.split_documents(docs)
    return chunks


def store_document(file_path, namespace):
    chunks = chunk_pdf(file_path)
    for i, chunk in enumerate(chunks):
        text = chunk.page_content
        text = ' '.join(text.split())#it splits text at any whitespace(space,newlines,..) and joins them together with one single spaces
        metadata = {"chunk_id":i, "file_name":os.path.basename(file_path),"text":text}
        vector = model_embeddings.embed_documents([text])[0]
        index.upsert([(f"{file_path}-chunk-{i}",vector,metadata)],namespace=namespace)


def retrieve_similar_chunks_and_answer(query, namespace, top_k=3):
    vector_query = model_embeddings.embed_query(query)
    results = index.query(vector=vector_query,top_k=top_k,include_metadata=True,namespace=namespace )#searching vector database for similar chunks, top k-3
    context = "\n".join([match.metadata['text'] for match in results.matches])#prepare context for the LLM
    prompt = prompt_template()
    formatting_prompt = prompt.format(context=context, question=query)
    answer = chat.predict(formatting_prompt)
    final_result = {"answer":answer,"sources":results.matches}
    print(f"\nAnswer: {final_result['answer']}")
    print(f"\nSources:")
    #match object ={id,score,metadata:filename,chunk_number,text}
    for result_number, match in enumerate(results.matches, 1):#results.match=list of matched objects(dictionary that have all informations about search result)
        print(f"\nSource {result_number}:")
        print(f"From file: {match.metadata['file_name']}")
        print(f"Chunk number: {match.metadata['chunk_id']}")
        print(f"How similar: {match.score:.4f}")
        print(f"Text: {match.metadata['text']}\n")        

#delete later
def test(query, namespace,top_k=3):
    vector_query = model_embeddings.embed_query(query)
    results = index.query(vector=vector_query,top_k=top_k,include_metadata=True, namespace=namespace)#searching vector database for similar chunks, top k-3
    print(results.matches)


if __name__=='__main__':
    file_path="./documents/attention_pdf.pdf"
    query = "Why does the Transformer use scaled dot-product attention instead of regular dot-product or additive attention?"
    namespace="attention_paper"
    #store_document(file_path,namespace)
    retrieve_similar_chunks_and_answer(query,namespace)


    #llm = ChatOpenAI(temperature=0, model_name='gpt-4o',openai_api_key=openai_key, openai_organization=openai_org)
    #chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    #while True:
     #   question = input('\n Ask a question: ')
      #  if question.lower()=='exit':
       #     break
        #answer = chain.invoke(question)
        #print('Answer:')
        #print(answer['result'])