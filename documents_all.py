from langchain_community.document_loaders import PyPDFLoader,UnstructuredWordDocumentLoader,UnstructuredPowerPointLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter,MarkdownHeaderTextSplitter
import json
import os

#Documents Parser Tool=chunking(["content":".....","section":"...."]) + embedding(with pgvector)
def load_documents(folder_path):
    all_docs = []
    file_names = []

    for filename in os.listdir(folder_path):#loop through all documents in
        file_path = os.path.join(folder_path,filename)
        if filename.endswith(".pdf"):
            docs = PyPDFLoader(file_path).load()#load and variable stores content of PDF
            splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)#split doc into chunks, each chunk is of size 600 and each new chunk repeats 100 characters fro the previous one(chunk)=>AI will remeber context better
        elif filename.endswith(".docx"):
            docs = UnstructuredWordDocumentLoader(file_path).load()#load and variable stores content of doc
            splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("##","section")])#splitting by headings(titles) and label chunk as section

        elif filename.endswith(".pptx"):
            docs = UnstructuredPowerPointLoader(file_path).load()#pptx format stores each slide speratly in the file, each slide represent a document
            splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)#if a slide exceed 200(chunk size) it splits slide into chunks

        elif filename.endswith(".txt"):
            with open(file_path, "r") as f:
                raw_text = f.read()
                docs = [Document(page_content=raw_text)]
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

        else:
            print("Unsupported file type")
            continue

        split_docs = splitter.split_documents(docs)
        all_docs.extend(split_docs)
        file_names.append(filename)
        
    return all_docs, file_names


if __name__=='__main__':
    folder_docs = "./documents"
    docs,names = load_documents(folder_docs)
    print(f"loaded {len(docs)} documents")
    for i, doc in enumerate(docs[:5]):
        print(f"chunk {i+1}:\n{doc.page_content[:300]}...")
