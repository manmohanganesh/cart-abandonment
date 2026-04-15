from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def load_documents(file_path):
    with open(file_path,"r",encoding="utf-8") as f:
        text = f.read()

    splitter = CharacterTextSplitter(chunk_size=100,chunk_overlap=20)
    docs = splitter.split_text(text)

    return docs

def create_vector_store(docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    
    vectorstore = Chroma.from_texts(docs,embedding=embeddings)

    return vectorstore