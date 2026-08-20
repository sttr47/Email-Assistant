import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

def get_llm():
    llm = ChatGroq(
        model_name=os.getenv("MODEL_NAME"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2, 
        streaming=True
    )
    return llm