from langchain_community.llms import CTransformers
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Configuration
model_file = "models/Votre_modele.gguf"


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
    prompt = PromptTemplate(template = template, input_variables=["question"])
    return prompt


# Creer une simple chaine
def creer_simple_chain(prompt, llm):
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain

# Commencer les tests

    # Creer une prompt, Il est recommande d'utiliser la template de modele
template = """<|im_start|>system
Vous êtes un assistant IA utile. Veuillez répondre aux utilisateurs avec précision.
<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant"""

prompt = creer_prompt(template)
llm = load_llm(model_file)
llm_chain = creer_simple_chain(prompt, llm)

question = "Combien font un plus un ?"
response = llm_chain.invoke({"question":question})
print(response)

