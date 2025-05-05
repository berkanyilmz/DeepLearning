from transformers import Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling, AutoProcessor, AutoModelForCausalLM
from preprocessing_dataset import format_data
from huggingface_hub import login
import json

def login_to_huggingface(token: str):
    print("Hugging Face bağlantısı kuruluyor...")
    login(token)

def load_tokenizer(model_name: str, cache_dir: str):
    print("Tokenizer yükleniyor...")
    tokenizer = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer yükleme tamamlandı!")
    return tokenizer

def load_model(model_name: str, cache_dir: str):
    print("Model yükleniyor...")
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
    print("Model yükleme tamamlandı!")
    return model

def load_and_format_data(json_path: str):
    with open(json_path, "r", encoding="utf8") as f:
        data = json.load(f)
    format_data(data)

def create_dataset(tokenizer, train_file: str):
    print("Veri seti yükleniyor...")
    return TextDataset(
        tokenizer=tokenizer,
        file_path=train_file,
        block_size=128,
    )

def get_data_collator(tokenizer):
    print("Data Collator oluşturuluyor...")
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

def get_training_args():
    print("Eğitim hiperparametreleri ayarlanıyor...")
    return TrainingArguments(
        output_dir="./output",
        logging_dir='./logs',
        logging_steps=20,
        overwrite_output_dir=True,
        num_train_epochs=12,
        per_device_train_batch_size=4,
        save_steps=100,
        save_total_limit=2,
        prediction_loss_only=True,
        fp16=True
    )

def train_model(model, tokenizer, train_dataset, data_collator, training_args):
    print("Eğitim başlatılıyor...")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        tokenizer=tokenizer
    )
    model.to("cuda")
    trainer.train()
    return model

def save_model(model, tokenizer, output_dir):
    print("Model kaydediliyor...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def generate_text(model, tokenizer, input_text: str):
    print("Metin üretiliyor...")
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(
        input_ids=inputs['input_ids'].to("cuda"),
        attention_mask=inputs["attention_mask"].to("cuda"),
        max_length=256,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9,
        pad_token_id=tokenizer.pad_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    HF_TOKEN = "your hugging face token"
    MODEL_NAME = "meta-llama/Llama-3.2-1B"
    CACHE_DIR = "E:/hugging face/llama"
    JSON_PATH = "tyt_turkish.json"
    TRAIN_FILE = "train_llama.txt"
    OUTPUT_DIR = "./Llama_question_generator"

    login_to_huggingface(HF_TOKEN)
    tokenizer = load_tokenizer(MODEL_NAME, CACHE_DIR)
    model = load_model(MODEL_NAME, CACHE_DIR)

    load_and_format_data(JSON_PATH)
    train_dataset = create_dataset(tokenizer, TRAIN_FILE)
    data_collator = get_data_collator(tokenizer)
    training_args = get_training_args()

    model = train_model(model, tokenizer, train_dataset, data_collator, training_args)
    save_model(model, tokenizer, OUTPUT_DIR)

    generated = generate_text(model, tokenizer, "soru üret")
    print(generated)