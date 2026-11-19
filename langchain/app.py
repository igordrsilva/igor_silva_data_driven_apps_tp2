from fastapi import FastAPI

from routes.Q1_simple_chatbot import router1
from routes.Q2_groq import router2
from routes.Q3_translate_text import router3

app = FastAPI(
    title="Teste de performance 2 - Desenvolvimento de Data-Driven Apps com Python", 
    description="Aplicação FastAPI integrada a diversos modelos de LLM do Hugging Face.", 
    version="1.0.0"
)
app.include_router(router1)
app.include_router(router2)
app.include_router(router3)