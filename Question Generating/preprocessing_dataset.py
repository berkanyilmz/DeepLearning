def format_data(data):
    # Veri setini gpt2 modeline uygun bir formata getirmek için kullanılır. Örnek olarak train.txt dosyasına bakılabilir
    option_letters = ['A', 'B', 'C', 'D', 'E']
    with open("train_llama.txt", "w", encoding="utf-8") as f:
        for item in data:
            f.write("<question>\n")
            f.write(f"Paragraf: {item['paragraph']}\n")
            f.write(f"Soru: {item['question']}\n")
            options = item["options"]

            for i, option in enumerate(options):
                f.write(f"{option_letters[i]}) {option}\n")

            f.write("\n")
            f.write("</question>\n")