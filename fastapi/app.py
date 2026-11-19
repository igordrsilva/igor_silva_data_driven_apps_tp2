from fastapi import FastAPI

from routes.Q1_text_generation_with_gpt2 import router1
from routes.Q2_text_translation import router2

app = FastAPI(
    title="Teste de performance 2 - Desenvolvimento de Data-Driven Apps com Python", 
    description="Aplicação FastAPI integrada a diversos modelos de LLM do Hugging Face.", 
    version="1.0.0"
)
app.include_router(router1)
app.include_router(router2)