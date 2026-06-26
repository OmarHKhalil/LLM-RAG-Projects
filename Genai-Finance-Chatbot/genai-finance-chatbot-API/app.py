from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from fastapi.responses import HTMLResponse
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# -------------------------------
# 1. Load Data & Prepare Vector Store
# -------------------------------

embeddings = HuggingFaceEmbeddings(model_name="path_to_local_model")

vectorstore = Chroma(
    collection_name="chatbot_finance",
    persist_directory="vectorstore",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# -------------------------------
# 2. API Keys & Models Configuration
# -------------------------------
GOOGLE_API_KEYS = [
    # Place your Gemini keys here if you have them.
]

MODEL_NAME = "gemini-2.5-flash"

OLLAMA_MODEL_NAME = "gemma4:e2b" 

def build_gemini(api_key):
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=api_key,
        max_retries=0
    )

def build_ollama():

    return ChatOllama(
        model=OLLAMA_MODEL_NAME,
        temperature=0.3
    )

# -------------------------------
# 3. RAG Chain Template
# -------------------------------

template = PromptTemplate(
    input_variables=["question", "context"],
    template=(
"""
You are the official AI Finance Expert for "Venezia Bank". 

- Your primary identity is "Venezia Bank Assistant". 
- You specialize in banking services (accounts, loans, cards) AND general financial securities (stocks, bonds, investment funds).

STRICT RULES:
1. LOYALTY: You only represent Venezia Bank. If a user asks about another bank (e.g., Al Rajhi, HSBC, etc.), respond with: 
   "I am the dedicated assistant for Venezia Bank. I cannot provide information or comparisons regarding other banks."
2. DOMAIN: Only answer questions related to finance, banking, or financial securities.
3. OUT-OF-DOMAIN: For non-financial questions (geography, history, etc.), respond with: 
   "Sorry, I can only help with financial and banking-related questions."
4. GROUNDEDNESS: Use the provided Context to answer. If the answer is not in the context, and it's a general finance question, answer concisely based on Venezia Bank's professional standards. If you don't know, say: "Sorry, I don't know."
5. GREETINGS: Be friendly and mention Venezia Bank (e.g., "Hello! Welcome to Venezia Bank.").

Question: {question}  
Context: {context}  
Answer:
"""
    )
)

def run_gemini_fallback(question):
    errors = []
    results_with_scores = vectorstore.similarity_search_with_score(question, k=1)
    
    if results_with_scores:
        doc, score = results_with_scores[0]
        print(f"\n Similarity Score (Distance): {score}") 
    else:
        print("\n No results found in Vector Store")
    
    if GOOGLE_API_KEYS:
        for api_key in GOOGLE_API_KEYS:
            try:
                llm = build_gemini(api_key)
                rag_chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | RunnableLambda(lambda x: print("\n📄 السياق:\n", x["context"]) or x)
                    | template
                    | llm
                    | StrOutputParser()
                )   
                res = rag_chain.invoke(question)
                return res
            except Exception as e:
                errors.append(f"Gemini Key Error: {str(e)}")
                continue

    print(f"\n🤖 Falling back to local Ollama model ({OLLAMA_MODEL_NAME})...")
    try:
        llm_local = build_ollama()
        rag_chain_local = (
            {"context": retriever, "question": RunnablePassthrough()}
            | RunnableLambda(lambda x: print("\n Context: \n", x["context"]) or x)
            | template
            | llm_local
            | StrOutputParser()
        )
        res = rag_chain_local.invoke(question)
        return res
    except Exception as local_e:
        errors.append(f"Ollama Local Error: {str(local_e)}")

    return {"error": "All API keys and local Ollama failed", "details": errors}

# -------------------------------
# 4. FastAPI
# -------------------------------
app = FastAPI(title="Genai Finance Chatbot")

class Query(BaseModel):
    question: str

@app.post("/genai-finance-chatbot")
def ask(query: Query):
    return run_gemini_fallback(query.question)

@app.post("/genai-finance-chatbot_html", response_class=HTMLResponse)
def ask_html(query: Query):
    response = run_gemini_fallback(query.question)
    text = response if isinstance(response, str) else str(response)

    lines = text.split("\n")
    html_lines = []
    in_ol = False
    in_ul = False

    for line in lines:
        stripped = line.strip()

        if stripped and stripped[0].isdigit() and stripped[1] == ".":
            if not in_ol:
                if in_ul:
                    html_lines.append("</ul>")
                    in_ul = False
                html_lines.append("<ol>")
                in_ol = True
            html_lines.append(f"<li>{stripped.split('.', 1)[1].strip()}</li>")

        elif stripped.startswith("- "):
            if not in_ul:
                if in_ol:
                    html_lines.append("</ol>")
                    in_ol = False
                html_lines.append("<ul>")
                in_ul = True
            html_lines.append(f"<li>{stripped[2:].strip()}</li>")

        else:
            if in_ol:
                html_lines.append("</ol>")
                in_ol = False
            if in_ul:
                html_lines.append("</ul>")
                in_ul = False
            if stripped:
                html_lines.append(f"<p>{stripped}</p>")

    if in_ol:
        html_lines.append("</ol>")
    if in_ul:
        html_lines.append("</ul>")

    return "\n".join(html_lines)