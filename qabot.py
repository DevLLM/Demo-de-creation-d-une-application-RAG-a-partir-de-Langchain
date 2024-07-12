from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS

# Configuration
model_file = "models/Votre_modele.gguf"
vector_db_path = "vectorstores/db_faiss"

# Charger LLM
def load_llm(model_file):
    llm = CTransformers(
        model=model_file,
        model_type="type de votre modele, ex: bert, gpt2, xlnet, roberta, distilbert, albert, etc.",
        max_new_tokens=1024,
        temperature=0.01 # 0.01 est un bon rapport qualite pour QA, max c'est 1
    )
    return llm

# Creer une prompt template
def creer_prompt(template):
    prompt = PromptTemplate(template = template, input_variables=["context", "question"])
    return prompt


# Creer une simple chaine
def creer_qa_chaine(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type= "stuff",
        retriever = db.as_retriever(search_kwargs = {"k":3}, max_tokens_limit=1024),
        return_source_documents = False,
        chain_type_kwargs= {'prompt': prompt}

    )
    return llm_chain

# Lire a partir de Vecteur DB
def lire_vectors_db():
    # Incorporation(Embedding)
    embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")
    db = FAISS.load_local(vector_db_path, embedding_model)
    return db


# Commencer les tests
db = lire_vectors_db()
llm = load_llm(model_file)

# Creer une prompt, Il est recommande d'utiliser la template de modele
template = """<|im_start|>system\nUtilisez les informations suivantes pour répondre à la question. Si vous ne connaissez pas la réponse, dites que vous ne savez pas, n'essayez pas d'inventer une réponse.\n
    {context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""
prompt = creer_prompt(template)

llm_chain  =creer_qa_chaine(prompt, llm, db)

# Executer la chaine
question = "La question est liée à quelque chose dans votre Donnes"
response = llm_chain.invoke({"query": question})
print(response)