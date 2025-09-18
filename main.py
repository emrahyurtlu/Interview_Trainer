import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import  HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

## LOADING
document_loader = TextLoader(file_path='questions.txt', encoding='utf-8')
documents = document_loader.load()

## SPLITTING
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splitted_documents = splitter.split_documents(documents)

## EMBEDDINGS
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#texts = [d.page_content for d in splitted_documents]  # <-- yalnızca metinleri al
#embedded_documents = embeddings.embed_documents(texts)
#print(embedded_documents[0])

## SAVING
vectorstore = FAISS.from_documents(splitted_documents, embeddings)
vectorstore.save_local("database")
print(vectorstore)

## QUERY THE DATA

## INTEGRATE WITH LLM

## OUTPUT FORMATTER

##INTEGRATE WITH UI