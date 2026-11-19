from fastapi import APIRouter, HTTPException
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from pydantic import BaseModel
from transformers import MarianMTModel, MarianTokenizer, pipeline

class TranslationRequest(BaseModel):
    text: str

router3 = APIRouter()

MODEL_NAME = "Helsinki-NLP/opus-mt-en-de"

tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME)

async def run_translation_inference(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


@router3.post("/traduzir-texto-en-de")
async def translate_text_endpoint(request: TranslationRequest) -> dict:
    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="O texto em inglês é obrigatório."
        )

    translated_text = await run_translation_inference(request.text)
    return {"translated_text": translated_text}