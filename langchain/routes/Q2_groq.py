import os
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("A variável GROQ_API_KEY não foi encontrada no arquivo .env.")

class TranslationRequest(BaseModel):
    text: str

llm = ChatGroq(
    model="openai/gpt-oss-20b", 
    temperature=0, 
    max_tokens=256
)

prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        "Você é um tradutor especializado de inglês para francês. "
        "Traduza o texto fornecido pelo usuário para o francês de forma natural e precisa. "
        "Retorne exclusivamente o texto traduzido, sem explicações ou comentários adicionais."
    ),
    ("human", "{text}")
])

translation_chain = prompt_template | llm | StrOutputParser()

async def run_translation_inference(text: str) -> str:
    return await translation_chain.ainvoke({"text": text})


router2 = APIRouter()

@router2.post("/traduzir-texto-en-fr")
async def translate_text_endpoint(request: TranslationRequest) -> dict:
    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="O texto em inglês é obrigatório."
        )

    translated_text = await run_translation_inference(request.text)
    return {"translated_text": translated_text}