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
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

## INTEGRATE WITH LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

## OUTPUT FORMATTER
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Give a clear and concise answer to the question using the following context::
Context: {context}
Question: {question}
Answer (Türkçe): 
"""
)

##INTEGRATE WITH UI
print("\n Q&A System Started! Type 'exit' to exit. \n")

while True:
    user_question = input("❓ Question: ")
    if user_question.lower() in ["exit", "quit", "çıkış"]:
        print("See you later!")
        break

    result = qa_chain.run(user_question)
    print(f"💡 Answer: {result}\n")
