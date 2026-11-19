from fastapi import APIRouter, HTTPException
from langchain_core.language_models import FakeListLLM
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel

class PromptRequest(BaseModel):
    prompt: str

fake_llm = FakeListLLM(
    responses=[
        "Python é uma linguagem de programação de alto nível, interpretada e amplamente usada em IA.",
        "LangChain é um framework para desenvolver aplicações baseadas em modelos de linguagem (LLMs).",
        "FastAPI é um framework web moderno e de alta performance para construir APIs em Python."
    ]
)

prompt_template = PromptTemplate.from_template(
    "Responda de forma objetiva à seguinte pergunta do usuário: {pergunta}"
)
chatbot_chain = prompt_template | fake_llm

router1 = APIRouter()

@router1.get("/")
async def root() -> dict:
    return {"message": "Hello FastAPI!"}

@router1.post("/chat")
async def chat_endpoint(request: PromptRequest) -> dict:
    if not request.prompt.strip():
        raise HTTPException(
            status_code=400,
            detail="O campo 'prompt' é obrigatório e não pode estar vazio."
        )

    resposta_simulada = chatbot_chain.invoke({"pergunta": request.prompt})

    return {
        "pergunta": request.prompt,
        "resposta": resposta_simulada
    }