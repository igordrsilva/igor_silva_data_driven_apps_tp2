from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from transformers import MarianTokenizer, MarianMTModel

class TranslationRequest(BaseModel):
    text: str

router2 = APIRouter()

MODEL_NAME = "Helsinki-NLP/opus-mt-en-fr"

tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME)

def run_translation_inference(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=256)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@router2.post("/traduzir-texto")
async def translate_text_endpoint(request: TranslationRequest):
    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="O texto em inglês é obrigatório."
        )

    translated_text = run_translation_inference(request.text)
    return {"translated_text": translated_text}