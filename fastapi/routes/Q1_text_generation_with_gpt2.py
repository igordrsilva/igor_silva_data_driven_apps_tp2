from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from transformers import pipeline, GenerationConfig

class PromptRequest(BaseModel):
    prompt: str

router1 = APIRouter()

pipe = pipeline(
    task="text-generation",
    model="pierreguillou/gpt2-small-portuguese"
)

generation_config = GenerationConfig(
    do_sample=True,
    max_new_tokens=50,
    temperature=0.6
)

def run_gpt2_inference(topic: str) -> str:
    prompt = f"Sobre {topic}, podemos dizer que..."

    resposta = pipe(
        prompt,
        generation_config=generation_config,
        return_full_text=False,
        clean_up_tokenization_spaces=False
    )

    return resposta[0]["generated_text"]

@router1.get("/")
async def root() -> dict:
    return {"message": "Hello FastAPI!"}

@router1.post("/gerar-uma-frase-com-gpt2")
async def generate_text_endpoint(api_text_input: str):
    if not api_text_input.strip():
        raise HTTPException(
            status_code=400,
            detail="Prompt is required."
        )

    generated_text = run_gpt2_inference(api_text_input)
    return {"generated_text": generated_text}
