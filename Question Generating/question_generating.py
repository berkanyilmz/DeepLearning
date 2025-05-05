from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def set_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(device, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

    return model, tokenizer

def generate_question(prompt:str, tokenizer, model, device):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=500,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9,
        pad_token_id=tokenizer.pad_token_id
    )

    soru = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Üretilen Soru:\n")
    print(soru)

device = set_device()
model, tokenizer = load_model(device, "./Llama_question_generator")
generate_question("soru üret", tokenizer, model, device)