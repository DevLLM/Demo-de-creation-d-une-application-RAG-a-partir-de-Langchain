from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings

# Declarer une variable
pdf_data_path = "data"
vector_db_path = "vectorstores/db_faiss"

# Fonction 1, Creer un vecteur DB a partir d'un texte
def creer_db_from_text():
    raw_text = """
    Votre Text
    """

    # Fractionner le texte
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=512,
        chunk_overlap=50,
        length_function=len

    )

    chunks = text_splitter.split_text(raw_text)

    # Embedding
    embedding_model = GPT4AllEmbeddings(model_file = "models/all-MiniLM-L6-v2-f16.gguf")

    # Base sur Faiss Vector DB
    db = FAISS.from_texts(texts=chunks, embedding=embedding_model)
    db.save_local(vector_db_path)
    return db

# Fonction 2, Creer un vecteur DB a partir d'un fichier pdf
def creer_db_from_files():
    # Declarer un chargeur pour analyser l'integralite du dossier Data
    loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls = PyPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Embedding
    embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(vector_db_path)
    return db


creer_db_from_files()

